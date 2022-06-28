/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <stdlib.h>  // qsort()

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "./vpx_scale_rtcd.h"

#include "vpx_dsp/bitreader_buffer.h"
#include "vpx_dsp/bitreader.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/mem_ops.h"
#include "vpx_scale/vpx_scale.h"
#include "vpx_util/vpx_thread.h"
#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
#include "vpx_util/vpx_debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG

#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_idct.h"
#include "vp9/common/vp9_thread_common.h"
#include "vp9/common/vp9_pred_common.h"
#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_reconintra.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_seg_common.h"
#include "vp9/common/vp9_tile_common.h"

#include "vp9/decoder/vp9_decodeframe.h"
#include <stdbool.h>

#include "vp9/decoder/vp9_detokenize.h"
#include "vp9/decoder/vp9_decodemv.h"
#include "vp9/decoder/vp9_decoder.h"
#include "vp9/decoder/vp9_dsubexp.h"
#include "vp9/decoder/vp9_job_queue.h"

#define MAX_VP9_HEADER_SIZE 80

MODE_INFO *set_offsets(VP9_COMMON *const cm, MACROBLOCKD *const xd, BLOCK_SIZE bsize, int mi_row, int mi_col, int bw, int bh,
                       int x_mis, int y_mis, int bwl, int bhl);

typedef int (*predict_recon_func)(TileWorkerData *twd, MODE_INFO *const mi,
                                  int plane, int row, int col, TX_SIZE tx_size);

typedef void (*intra_recon_func)(TileWorkerData *twd, MODE_INFO *const mi,
                                 int plane, int row, int col, TX_SIZE tx_size);

static int read_is_valid(const uint8_t *start, size_t len, const uint8_t *end) {
  return len != 0 && len <= (size_t)(end - start);
}

static int decode_unsigned_max(struct vpx_read_bit_buffer *rb, int max) {
  const int data = vpx_rb_read_literal(rb, get_unsigned_bits(max));
  return data > max ? max : data;
}

static TX_MODE read_tx_mode(vpx_reader *r) {
  TX_MODE tx_mode = vpx_read_literal(r, 2);
  if (tx_mode == ALLOW_32X32) tx_mode += vpx_read_bit(r);
  return tx_mode;
}

static void read_tx_mode_probs(struct tx_probs *tx_probs, vpx_reader *r) {
  int i, j;

  for (i = 0; i < TX_SIZE_CONTEXTS; ++i)
    for (j = 0; j < TX_SIZES - 3; ++j)
      vp9_diff_update_prob(r, &tx_probs->p8x8[i][j]);

  for (i = 0; i < TX_SIZE_CONTEXTS; ++i)
    for (j = 0; j < TX_SIZES - 2; ++j)
      vp9_diff_update_prob(r, &tx_probs->p16x16[i][j]);

  for (i = 0; i < TX_SIZE_CONTEXTS; ++i)
    for (j = 0; j < TX_SIZES - 1; ++j)
      vp9_diff_update_prob(r, &tx_probs->p32x32[i][j]);
}

static void read_switchable_interp_probs(FRAME_CONTEXT *fc, vpx_reader *r) {
  int i, j;
  for (j = 0; j < SWITCHABLE_FILTER_CONTEXTS; ++j)
    for (i = 0; i < SWITCHABLE_FILTERS - 1; ++i)
      vp9_diff_update_prob(r, &fc->switchable_interp_prob[j][i]);
}

static void read_inter_mode_probs(FRAME_CONTEXT *fc, vpx_reader *r) {
  int i, j;
  for (i = 0; i < INTER_MODE_CONTEXTS; ++i)
    for (j = 0; j < INTER_MODES - 1; ++j)
      vp9_diff_update_prob(r, &fc->inter_mode_probs[i][j]);
}

static REFERENCE_MODE read_frame_reference_mode(const VP9_COMMON *cm,
                                                vpx_reader *r) {
  if (vp9_compound_reference_allowed(cm)) {
    return vpx_read_bit(r)
               ? (vpx_read_bit(r) ? REFERENCE_MODE_SELECT : COMPOUND_REFERENCE)
               : SINGLE_REFERENCE;
  } else {
    return SINGLE_REFERENCE;
  }
}

static void read_frame_reference_mode_probs(VP9_COMMON *cm, vpx_reader *r) {
  FRAME_CONTEXT *const fc = cm->fc;
  int i;

  if (cm->reference_mode == REFERENCE_MODE_SELECT)
    for (i = 0; i < COMP_INTER_CONTEXTS; ++i)
      vp9_diff_update_prob(r, &fc->comp_inter_prob[i]);

  if (cm->reference_mode != COMPOUND_REFERENCE)
    for (i = 0; i < REF_CONTEXTS; ++i) {
      vp9_diff_update_prob(r, &fc->single_ref_prob[i][0]);
      vp9_diff_update_prob(r, &fc->single_ref_prob[i][1]);
    }

  if (cm->reference_mode != SINGLE_REFERENCE)
    for (i = 0; i < REF_CONTEXTS; ++i)
      vp9_diff_update_prob(r, &fc->comp_ref_prob[i]);
}

static void update_mv_probs(vpx_prob *p, int n, vpx_reader *r) {
  int i;
  for (i = 0; i < n; ++i)
    if (vpx_read(r, MV_UPDATE_PROB)) p[i] = (vpx_read_literal(r, 7) << 1) | 1;
}

static void read_mv_probs(nmv_context *ctx, int allow_hp, vpx_reader *r) {
  int i, j;

  update_mv_probs(ctx->joints, MV_JOINTS - 1, r);

  for (i = 0; i < 2; ++i) {
    nmv_component *const comp_ctx = &ctx->comps[i];
    update_mv_probs(&comp_ctx->sign, 1, r);
    update_mv_probs(comp_ctx->classes, MV_CLASSES - 1, r);
    update_mv_probs(comp_ctx->class0, CLASS0_SIZE - 1, r);
    update_mv_probs(comp_ctx->bits, MV_OFFSET_BITS, r);
  }

  for (i = 0; i < 2; ++i) {
    nmv_component *const comp_ctx = &ctx->comps[i];
    for (j = 0; j < CLASS0_SIZE; ++j)
      update_mv_probs(comp_ctx->class0_fp[j], MV_FP_SIZE - 1, r);
    update_mv_probs(comp_ctx->fp, 3, r);
  }

  if (allow_hp) {
    for (i = 0; i < 2; ++i) {
      nmv_component *const comp_ctx = &ctx->comps[i];
      update_mv_probs(&comp_ctx->class0_hp, 1, r);
      update_mv_probs(&comp_ctx->hp, 1, r);
    }
  }
}

static void inverse_transform_block_inter(MACROBLOCKD *xd, int plane,
                                          const TX_SIZE tx_size,
                                          tran_high_t *buf,
                                          int stride, int eob, tran_low_t* dq) {
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = dq;
  assert(eob > 0);
#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    if (xd->lossless) {
      vpx_highbd_iwht4x4_16_add_c(dqcoeff, buf, stride, xd->bd);
    } else {
      switch (tx_size) {
        case TX_4X4:
          vpx_highbd_idct4x4_16_add_c(dqcoeff, buf, stride, xd->bd);
          break;
        case TX_8X8:
          vpx_highbd_idct8x8_64_add_c(dqcoeff, buf, stride, xd->bd);
          break;
        case TX_16X16:
          vpx_highbd_idct16x16_256_add_c(dqcoeff, buf, stride, xd->bd);
          break;
        case TX_32X32:
          vpx_highbd_idct32x32_1024_add_c(dqcoeff, buf, stride, xd->bd);
          break;
        default: assert(0 && "Invalid transform size");
      }
    }
  } else {
    if (xd->lossless) {
      vp9_iwht4x4_add(dqcoeff, buf, stride, eob);
    } else {
      switch (tx_size) {
        case TX_4X4: vp9_idct4x4_add(dqcoeff, buf, stride, eob); break;
        case TX_8X8: vp9_idct8x8_add(dqcoeff, buf, stride, eob); break;
        case TX_16X16: vp9_idct16x16_add(dqcoeff, buf, stride, eob); break;
        case TX_32X32: vp9_idct32x32_add(dqcoeff, buf, stride, eob); break;
        default: assert(0 && "Invalid transform size"); return;
      }
    }
  }
#else
  if (xd->lossless) {
    vp9_iwht4x4_add(dqcoeff, dst, stride, eob);
  } else {
    switch (tx_size) {
      case TX_4X4: vp9_idct4x4_add(dqcoeff, dst, stride, eob); break;
      case TX_8X8: vp9_idct8x8_add(dqcoeff, dst, stride, eob); break;
      case TX_16X16: vp9_idct16x16_add(dqcoeff, dst, stride, eob); break;
      case TX_32X32: vp9_idct32x32_add(dqcoeff, dst, stride, eob); break;
      default: assert(0 && "Invalid transform size"); return;
    }
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static void inverse_transform_block_intra(MACROBLOCKD *xd, int plane,
                                          const TX_TYPE tx_type,
                                          const TX_SIZE tx_size,
                                          tran_high_t *buf,
                                          int stride, int eob, tran_low_t* dq) {
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = dq;
  assert(eob > 0);
#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    if (xd->lossless) {
      vpx_highbd_iwht4x4_16_add_c(dqcoeff, buf, stride, xd->bd);
    } else {
      switch (tx_size) {
        case TX_4X4:
          vp9_highbd_iht4x4_16_add_c(dqcoeff, buf, stride, tx_type, xd->bd);
          break;
        case TX_8X8:
          vp9_highbd_iht8x8_64_add_c(dqcoeff, buf, stride, tx_type, xd->bd);
          break;
        case TX_16X16:
          vp9_highbd_iht16x16_256_add_c(dqcoeff, buf, stride, tx_type, xd->bd);
          break;
        case TX_32X32:
          vpx_highbd_idct32x32_1024_add_c(dqcoeff, buf, stride, xd->bd);
          break;
        default: assert(0 && "Invalid transform size");
      }
    }
  } else {
    if (xd->lossless) {
      vp9_iwht4x4_add(dqcoeff, buf, stride, eob);
    } else {
      switch (tx_size) {
        case TX_4X4: vp9_iht4x4_add(tx_type, dqcoeff, buf, stride, eob); break;
        case TX_8X8: vp9_iht8x8_add(tx_type, dqcoeff, buf, stride, eob); break;
        case TX_16X16:
          vp9_iht16x16_add(tx_type, dqcoeff, buf, stride, eob);
          break;
        case TX_32X32: vp9_idct32x32_add(dqcoeff, buf, stride, eob); break;
        default: assert(0 && "Invalid transform size"); return;
      }
    }
  }
#else
  if (xd->lossless) {
    vp9_iwht4x4_add(dqcoeff, dst, stride, eob);
  } else {
    switch (tx_size) {
      case TX_4X4: vp9_iht4x4_add(tx_type, dqcoeff, dst, stride, eob); break;
      case TX_8X8: vp9_iht8x8_add(tx_type, dqcoeff, dst, stride, eob); break;
      case TX_16X16:
        vp9_iht16x16_add(tx_type, dqcoeff, dst, stride, eob);
        break;
      case TX_32X32: vp9_idct32x32_add(dqcoeff, dst, stride, eob); break;
      default: assert(0 && "Invalid transform size"); return;
    }
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static void block_sum(MACROBLOCKD *xd, uint8_t *dst, tran_high_t *buf,
                      TX_SIZE tx_size, int stride) {
  uint16_t *const dst16 = CONVERT_TO_SHORTPTR(dst);
  if (xd->lossless) {
    for (int i = 0; i < 4; i++) {
      dst16[stride * 0] = highbd_clip_pixel_add(
          dst16[stride * 0], HIGHBD_WRAPLOW(buf[0], xd->bd), xd->bd);
      dst16[stride * 1] = highbd_clip_pixel_add(
          dst16[stride * 1], HIGHBD_WRAPLOW(buf[1], xd->bd), xd->bd);
      dst16[stride * 2] = highbd_clip_pixel_add(
          dst16[stride * 2], HIGHBD_WRAPLOW(buf[2], xd->bd), xd->bd);
      dst16[stride * 3] = highbd_clip_pixel_add(
          dst16[stride * 3], HIGHBD_WRAPLOW(buf[3], xd->bd), xd->bd);
    }
  } else {
    switch (tx_size) {
      case TX_4X4:
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            dst16[j * stride + i] = highbd_clip_pixel_add(
                dst16[j * stride + i], buf[j * stride + i], xd->bd);
          }
        }
        break;
      case TX_8X8:
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            dst16[j * stride + i] = highbd_clip_pixel_add(
                dst16[j * stride + i], buf[j * stride + i], xd->bd);
          }
        }
        break;
      case TX_16X16:
        for (int i = 0; i < 16; ++i) {
          for (int j = 0; j < 16; ++j) {
            dst16[j * stride + i] = highbd_clip_pixel_add(
                dst16[j * stride + i], buf[j * stride + i], xd->bd);
          }
        }
        break;
      case TX_32X32:
        for (int i = 0; i < 32; ++i) {
          for (int j = 0; j < 32; ++j) {
            dst16[j * stride + i] = highbd_clip_pixel_add(
                dst16[j * stride + i], buf[j * stride + i], xd->bd);
          }
        }
        break;
      default: assert(0 && "Invalid transform size");
    }
  }
}

static void parse_intra_block_row_mt(TileWorkerData *twd, MODE_INFO *const mi,
                                     int plane, int row, int col,
                                     TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &twd->xd;
  PREDICTION_MODE mode = (plane == 0) ? mi->mode : mi->uv_mode;

  if (mi->sb_type < BLOCK_8X8)
    if (plane == 0) mode = xd->mi[0]->bmi[(row << 1) + col].as_mode;

  if (!mi->skip) {
    struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_TYPE tx_type =
        (plane || xd->lossless) ? DCT_DCT : intra_mode_to_tx_type_lookup[mode];
    const scan_order *sc = (plane || xd->lossless)
                               ? &vp9_default_scan_orders[tx_size]
                               : &vp9_scan_orders[tx_size][tx_type];
    *pd->eob = vp9_decode_block_tokens(twd, plane, sc, col, row, tx_size,
                                       mi->segment_id);
    /* Keep the alignment to 16 */
    pd->dqcoeff += (16 << (tx_size << 1));
    pd->eob++;
  }
}

static void predict_and_reconstruct_intra_block_row_mt(TileWorkerData *twd,
                                                       MODE_INFO *const mi,
                                                       int plane, int row,
                                                       int col,
                                                       TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &twd->xd;
  struct macroblockd_plane *const pd = &xd->plane[plane];
  PREDICTION_MODE mode = (plane == 0) ? mi->mode : mi->uv_mode;
  uint8_t *dst = &pd->dst.buf[4 * row * pd->dst.stride + 4 * col];

  if (mi->sb_type < BLOCK_8X8)
    if (plane == 0) mode = xd->mi[0]->bmi[(row << 1) + col].as_mode;

  vp9_predict_intra_block(xd, pd->n4_wl, tx_size, mode, dst, pd->dst.stride,
                          dst, pd->dst.stride, col, row, plane);

  if (!mi->skip) {
    const TX_TYPE tx_type =
        (plane || xd->lossless) ? DCT_DCT : intra_mode_to_tx_type_lookup[mode];
    if (*pd->eob > 0) {
      inverse_transform_block_intra(xd, plane, tx_type, tx_size, dst,
                                    pd->dst.stride, *pd->eob, 0);
    }
    /* Keep the alignment to 16 */
    pd->dqcoeff += (16 << (tx_size << 1));
    pd->eob++;
  }
}


static int parse_inter_block_row_mt(TileWorkerData *twd, MODE_INFO *const mi,
                                    int plane, int row, int col,
                                    TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &twd->xd;
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const scan_order *sc = &vp9_default_scan_orders[tx_size];
  const int eob = vp9_decode_block_tokens(twd, plane, sc, col, row, tx_size,
                                          mi->segment_id);

  *pd->eob = eob;
  pd->dqcoeff += (16 << (tx_size << 1));
  pd->eob++;

  return eob;
}

static int reconstruct_inter_block_row_mt(TileWorkerData *twd,
                                          MODE_INFO *const mi, int plane,
                                          int row, int col, TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &twd->xd;
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const int eob = *pd->eob;

  (void)mi;
  if (eob > 0) {
    inverse_transform_block_inter(
        xd, plane, tx_size, &pd->dst.buf[4 * row * pd->dst.stride + 4 * col],
        pd->dst.stride, eob, 0);
  }
  pd->dqcoeff += (16 << (tx_size << 1));
  pd->eob++;

  return eob;
}

static void build_mc_border(const uint8_t *src, int src_stride, uint8_t *dst,
                            int dst_stride, int x, int y, int b_w, int b_h,
                            int w, int h) {
  // Get a pointer to the start of the real data for this row.
  const uint8_t *ref_row = src - x - y * src_stride;

  if (y >= h)
    ref_row += (h - 1) * src_stride;
  else if (y > 0)
    ref_row += y * src_stride;

  do {
    int right = 0, copy;
    int left = x < 0 ? -x : 0;

    if (left > b_w) left = b_w;

    if (x + b_w > w) right = x + b_w - w;

    if (right > b_w) right = b_w;

    copy = b_w - left - right;

    if (left) memset(dst, ref_row[0], left);

    if (copy) memcpy(dst + left, ref_row + x + left, copy);

    if (right) memset(dst + left + copy, ref_row[w - 1], right);

    dst += dst_stride;
    ++y;

    if (y > 0 && y < h) ref_row += src_stride;
  } while (--b_h);
}

#if CONFIG_VP9_HIGHBITDEPTH
static void high_build_mc_border(const uint8_t *src8, int src_stride,
                                 uint16_t *dst, int dst_stride, int x, int y,
                                 int b_w, int b_h, int w, int h) {
  // Get a pointer to the start of the real data for this row.
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref_row = src - x - y * src_stride;

  if (y >= h)
    ref_row += (h - 1) * src_stride;
  else if (y > 0)
    ref_row += y * src_stride;

  do {
    int right = 0, copy;
    int left = x < 0 ? -x : 0;

    if (left > b_w) left = b_w;

    if (x + b_w > w) right = x + b_w - w;

    if (right > b_w) right = b_w;

    copy = b_w - left - right;

    if (left) vpx_memset16(dst, ref_row[0], left);

    if (copy) memcpy(dst + left, ref_row + x + left, copy * sizeof(uint16_t));

    if (right) vpx_memset16(dst + left + copy, ref_row[w - 1], right);

    dst += dst_stride;
    ++y;

    if (y > 0 && y < h) ref_row += src_stride;
  } while (--b_h);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if CONFIG_VP9_HIGHBITDEPTH
static void extend_and_predict(const uint8_t *buf_ptr1,
                               int pre_buf_stride, int x0, int y0, int b_w,
                               int b_h, int frame_width, int frame_height,
                               int border_offset, uint8_t *const dst,
                               int dst_buf_stride, int subpel_x, int subpel_y,
                               const InterpKernel *kernel,
                               const struct scale_factors *sf, int bd,
                               int w, int h, int ref, int xs, int ys) {
  //uint16_t *mc_buf_high = twd->extend_and_predict_buf;
  DECLARE_ALIGNED(16, uint16_t, mc_buf_high[80 * 2 * 80 * 2]);
  high_build_mc_border(buf_ptr1, pre_buf_stride, mc_buf_high, b_w, x0, y0, b_w, b_h, frame_width, frame_height);
  highbd_inter_predictor(mc_buf_high + border_offset, b_w, CONVERT_TO_SHORTPTR(dst), dst_buf_stride, subpel_x,
                             subpel_y, sf, w, h, ref, kernel, xs, ys, bd);
  //if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
  //  high_build_mc_border(buf_ptr1, pre_buf_stride, mc_buf_high, b_w, x0, y0,
  //                       b_w, b_h, frame_width, frame_height);
  //  highbd_inter_predictor(mc_buf_high + border_offset, b_w,
  //                         CONVERT_TO_SHORTPTR(dst), dst_buf_stride, subpel_x,
  //                         subpel_y, sf, w, h, ref, kernel, xs, ys, xd->bd);
  //} else {
  //  build_mc_border(buf_ptr1, pre_buf_stride, (uint8_t *)mc_buf_high, b_w, x0,
  //                  y0, b_w, b_h, frame_width, frame_height);
  //  inter_predictor(((uint8_t *)mc_buf_high) + border_offset, b_w, dst,
  //                  dst_buf_stride, subpel_x, subpel_y, sf, w, h, ref, kernel,
  //                  xs, ys);
  //}
}
#else
static void extend_and_predict(TileWorkerData *twd, const uint8_t *buf_ptr1,
                               int pre_buf_stride, int x0, int y0, int b_w,
                               int b_h, int frame_width, int frame_height,
                               int border_offset, uint8_t *const dst,
                               int dst_buf_stride, int subpel_x, int subpel_y,
                               const InterpKernel *kernel,
                               const struct scale_factors *sf, int w, int h,
                               int ref, int xs, int ys) {
  uint8_t *mc_buf = (uint8_t *)twd->extend_and_predict_buf;
  const uint8_t *buf_ptr;

  build_mc_border(buf_ptr1, pre_buf_stride, mc_buf, b_w, x0, y0, b_w, b_h,
                  frame_width, frame_height);
  buf_ptr = mc_buf + border_offset;

  inter_predictor(buf_ptr, b_w, dst, dst_buf_stride, subpel_x, subpel_y, sf, w,
                  h, ref, kernel, xs, ys);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void dec_build_inter_predictors(IntraBuf temp, int bw, int bh, int x, int y, int w, int h, int mi_x, int mi_y,
                                       const InterpKernel *kernel, const struct scale_factors *sf, int buf_stride, uint8_t *dst_buf, const MV *mv,
                                       uint8_t *ref_frame, int frame_width, int frame_height, int is_scaled, int ref) {
  uint8_t *const dst = dst_buf + buf_stride * y + x;
  MV32 scaled_mv;
  int xs, ys, x0, y0, x0_16, y0_16, subpel_x, subpel_y;
  uint8_t *buf_ptr;

  if (is_scaled) {
    const int spel_left = (VP9_INTERP_EXTEND + bw) << SUBPEL_BITS;
    const int spel_right = spel_left - SUBPEL_SHIFTS;
    const int spel_top = (VP9_INTERP_EXTEND + bh) << SUBPEL_BITS;
    const int spel_bottom = spel_top - SUBPEL_SHIFTS;
    MV mv_q4 = { (short)(mv->row * (1 << (1 - temp.t_subsampling_y))),
                 (short)(mv->col * (1 << (1 - temp.t_subsampling_x))) };
    assert(temp.t_subsampling_x <= 1);
    assert(temp.t_subsampling_y <= 1);

    clamp_mv( &mv_q4,*temp.mb_to_left_edge * (1 << (1 - temp.t_subsampling_x)) - spel_left,
        *temp.mb_to_right_edge * (1 << (1 - temp.t_subsampling_x)) + spel_right,
        *temp.mb_to_top_edge * (1 << (1 - temp.t_subsampling_y)) - spel_top,
        *temp.mb_to_bottom_edge * (1 << (1 - temp.t_subsampling_y)) + spel_bottom);

    // Co-ordinate of containing block to pixel precision.
    int x_start = (-*temp.mb_to_left_edge >> (3 + temp.t_subsampling_x));
    int y_start = (-*temp.mb_to_top_edge >> (3 + temp.t_subsampling_y));
#if 0  // CONFIG_BETTER_HW_COMPATIBILITY
    assert(xd->mi[0]->sb_type != BLOCK_4X8 &&
           xd->mi[0]->sb_type != BLOCK_8X4);
    assert(mv_q4.row == mv->row * (1 << (1 - pd->subsampling_y)) &&
           mv_q4.col == mv->col * (1 << (1 - pd->subsampling_x)));
#endif
    // Co-ordinate of the block to 1/16th pixel precision.
    x0_16 = (x_start + x) << SUBPEL_BITS;
    y0_16 = (y_start + y) << SUBPEL_BITS;

    // Co-ordinate of current block in reference frame
    // to 1/16th pixel precision.
    x0_16 = sf->scale_value_x(x0_16, sf);
    y0_16 = sf->scale_value_y(y0_16, sf);

    // Map the top left corner of the block into the reference frame.
    x0 = sf->scale_value_x(x_start + x, sf);
    y0 = sf->scale_value_y(y_start + y, sf);

    // Scale the MV and incorporate the sub-pixel offset of the block
    // in the reference frame.
    scaled_mv = vp9_scale_mv(&mv_q4, mi_x + x, mi_y + y, sf);
    xs = sf->x_step_q4;
    ys = sf->y_step_q4;
  } else {
    // Co-ordinate of containing block to pixel precision.
    x0 = (-*temp.mb_to_left_edge >> (3 + temp.t_subsampling_x)) + x;
    y0 = (-*temp.mb_to_top_edge >> (3 + temp.t_subsampling_y)) + y;

    // Co-ordinate of the block to 1/16th pixel precision.
    x0_16 = x0 << SUBPEL_BITS;
    y0_16 = y0 << SUBPEL_BITS;

    scaled_mv.row = mv->row * (1 << (1 - temp.t_subsampling_y));
    scaled_mv.col = mv->col * (1 << (1 - temp.t_subsampling_x));
    xs = ys = 16;
  }
  subpel_x = scaled_mv.col & SUBPEL_MASK;
  subpel_y = scaled_mv.row & SUBPEL_MASK;

  // Calculate the top left corner of the best matching block in the
  // reference frame.
  x0 += scaled_mv.col >> SUBPEL_BITS;
  y0 += scaled_mv.row >> SUBPEL_BITS;
  x0_16 += scaled_mv.col;
  y0_16 += scaled_mv.row;

  // Get reference block pointer.
  buf_ptr = ref_frame + y0 * buf_stride + x0;

  // Do border extension if there is motion or the
  // width/height is not a multiple of 8 pixels.
  if (is_scaled || scaled_mv.col || scaled_mv.row || (frame_width & 0x7) ||
      (frame_height & 0x7)) {
    int y1 = ((y0_16 + (h - 1) * ys) >> SUBPEL_BITS) + 1;

    // Get reference block bottom right horizontal coordinate.
    int x1 = ((x0_16 + (w - 1) * xs) >> SUBPEL_BITS) + 1;
    int x_pad = 0, y_pad = 0;

    if (subpel_x || (sf->x_step_q4 != SUBPEL_SHIFTS)) {
      x0 -= VP9_INTERP_EXTEND - 1;
      x1 += VP9_INTERP_EXTEND;
      x_pad = 1;
    }

    if (subpel_y || (sf->y_step_q4 != SUBPEL_SHIFTS)) {
      y0 -= VP9_INTERP_EXTEND - 1;
      y1 += VP9_INTERP_EXTEND;
      y_pad = 1;
    }

    // Skip border extension if block is inside the frame.
    if (x0 < 0 || x0 > frame_width - 1 || x1 < 0 || x1 > frame_width - 1 ||
        y0 < 0 || y0 > frame_height - 1 || y1 < 0 || y1 > frame_height - 1) {
      // Extend the border.
      const uint8_t *const buf_ptr1 = ref_frame + y0 * buf_stride + x0;
      const int b_w = x1 - x0 + 1;
      const int b_h = y1 - y0 + 1;
      const int border_offset = y_pad * 3 * b_w + x_pad * 3;

      extend_and_predict(buf_ptr1, buf_stride, x0, y0, b_w, b_h, frame_width,
                         frame_height, border_offset, dst, buf_stride,
                         subpel_x, subpel_y, kernel, sf, temp.bit_depth,
#if CONFIG_VP9_HIGHBITDEPTH
                         w,
#endif
                         h, ref, xs, ys);
      return;
    }
  }

  highbd_inter_predictor(CONVERT_TO_SHORTPTR(buf_ptr), buf_stride,
                         CONVERT_TO_SHORTPTR(dst), buf_stride, subpel_x,
                         subpel_y, sf, w, h, ref, kernel, xs, ys, temp.bit_depth);
//#if CONFIG_VP9_HIGHBITDEPTH
//  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
//    highbd_inter_predictor(CONVERT_TO_SHORTPTR(buf_ptr), buf_stride,
//                           CONVERT_TO_SHORTPTR(dst), dst_buf->stride, subpel_x,
//                           subpel_y, sf, w, h, ref, kernel, xs, ys, temp->bd);
//  } else {
//    inter_predictor(buf_ptr, buf_stride, dst, dst_buf->stride, subpel_x,
//                    subpel_y, sf, w, h, ref, kernel, xs, ys);
//  }
//#else
//  inter_predictor(buf_ptr, buf_stride, dst, dst_buf->stride, subpel_x, subpel_y,
//                  sf, w, h, ref, kernel, xs, ys);
//#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static void set_plane_n4(MACROBLOCKD *const xd, int bw, int bh, int bwl,
                         int bhl) {
  int i;
  for (i = 0; i < MAX_MB_PLANE; i++) {
    xd->plane[i].n4_w = (bw << 1) >> xd->plane[i].subsampling_x;
    xd->plane[i].n4_h = (bh << 1) >> xd->plane[i].subsampling_y;
    xd->plane[i].n4_wl = bwl - xd->plane[i].subsampling_x;
    xd->plane[i].n4_hl = bhl - xd->plane[i].subsampling_y;
  }
}

static void dec_build_inter_predictors_sb(FrameInformation fi, IntraBuf temp, uint8_t *alloc,
                                          uint8_t *frame_refs[3], struct scale_factors *const sf[3]) {
  int mi_row = temp.block_settings[3];
  int mi_col = temp.block_settings[5];
  const int bh = 1 << (*temp.bhl - 1);
  const int bw = 1 << (*temp.bwl - 1);
  int size = 0;
  
  uint8_t *y_buf = (uint8_t *)yv12_align_addr(alloc + (fi.border * fi.y_stride) + fi.border, fi.vp9_byte_align);
  uint8_t *u_buf = (uint8_t *)yv12_align_addr(alloc + fi.yplane_size + (fi.uv_border_h * fi.uv_stride) + fi.uv_border_w, fi.vp9_byte_align);
  uint8_t *v_buf = (uint8_t *)yv12_align_addr(alloc + fi.yplane_size + fi.uvplane_size + (fi.uv_border_h * fi.uv_stride) + fi.uv_border_w, fi.vp9_byte_align);

  uint8_t *const buffers[MAX_MB_PLANE] = { y_buf, u_buf, v_buf };
  const int strides[MAX_MB_PLANE] = { fi.y_stride, fi.uv_stride, fi.uv_stride };

  int plane;
  const int mi_x = mi_col * MI_SIZE;
  const int mi_y = mi_row * MI_SIZE;
  const InterpKernel *kernel = vp9_filter_kernels[temp.block_settings[8]];
  int ref;
  int is_scaled;
  
  for (ref = 0; ref < 1 + temp.block_settings[7]; ++ref) {
    int k = temp.ref_frame[ref] - LAST_FRAME;
    uint8_t *refalloc = frame_refs[k];
#if CONFIG_VP9_HIGHBITDEPTH
    refalloc = CONVERT_TO_BYTEPTR(refalloc);
#endif
    uint8_t *ref_y_buf = (uint8_t *)yv12_align_addr(refalloc + (fi.border * fi.y_stride) + fi.border, fi.vp9_byte_align);
    uint8_t *ref_u_buf = (uint8_t *)yv12_align_addr(refalloc + fi.yplane_size + (fi.uv_border_h * fi.uv_stride) + fi.uv_border_w, fi.vp9_byte_align);
    uint8_t *ref_v_buf = (uint8_t *)yv12_align_addr(refalloc + fi.yplane_size + fi.uvplane_size + (fi.uv_border_h * fi.uv_stride) + fi.uv_border_w, fi.vp9_byte_align);
    uint8_t *const ref_buffers[MAX_MB_PLANE] = { ref_y_buf, ref_u_buf, ref_v_buf };

    is_scaled = vp9_is_scaled(sf[k]);

    if (*temp.sb_type < BLOCK_8X8) {
      for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
        uint8_t *dst_buf =  buffers[plane] + scaled_buffer_offset((MI_SIZE * mi_col) >> temp.t_subsampling_x,
                                 (MI_SIZE * mi_row) >> temp.t_subsampling_y, strides[plane], NULL);
      
        const int num_4x4_w = (bw << 1) >> temp.t_subsampling_x;
        const int num_4x4_h = (bh << 1) >> temp.t_subsampling_y;
        const int n4w_x4 = 4 * num_4x4_w;
        const int n4h_x4 = 4 * num_4x4_h;
        const int buf_stride = strides[plane];
        int i = 0, x, y;
        for (y = 0; y < num_4x4_h; ++y) {
          for (x = 0; x < num_4x4_w; ++x) {
            const MV mv = { temp.mv[size], temp.mv[size + 2] };
            ++size;
            ++i;
            int frame_width;
            int frame_height;
            uint8_t *ref_frame;
      
            if (plane == 0) {
              frame_width = fi.y_crop_width;
              frame_height = fi.y_crop_height;
            } else {
              frame_width = fi.uv_crop_width;
              frame_height = fi.uv_crop_height;
            }
            ref_frame = ref_buffers[plane];
      
            dec_build_inter_predictors(temp, n4w_x4, n4h_x4, 4 * x, 4 * y, 4, 4, mi_x, mi_y, kernel,
                sf[k], buf_stride, dst_buf, &mv, ref_frame, frame_width, frame_height, is_scaled, ref);
          }
        }
      }
    } else {
      const MV mv = { temp.mv[size], temp.mv[size + 2] };
      ++size;
      temp.t_subsampling_x = temp.block_settings[5];
      temp.t_subsampling_y = temp.block_settings[6];
      
      uint8_t* dst_buf = buffers[temp.block_settings[2]] + scaled_buffer_offset((MI_SIZE * mi_col) >> temp.t_subsampling_x, 
          (MI_SIZE * mi_row) >> temp.t_subsampling_y, strides[temp.block_settings[2]], NULL);
      
      const int buf_stride = strides[temp.block_settings[2]];

      int frame_width;
      int frame_height;
      uint8_t* ref_frame;
      
      if (temp.block_settings[2] == 0) {
        frame_width = fi.y_crop_width;
        frame_height = fi.y_crop_height;
      } else {
        frame_width = fi.uv_crop_width;
        frame_height = fi.uv_crop_height;
      }
      ref_frame = ref_buffers[temp.block_settings[2]];
      
      dec_build_inter_predictors(temp, 8, 8, 8 * temp.block_settings[0], 8 * temp.block_settings[1], 8, 8, mi_x, mi_y, kernel, sf[k], 
          buf_stride, dst_buf, &mv, ref_frame, frame_width, frame_height, is_scaled, ref);
    }
  }
}

static INLINE void dec_reset_skip_context(MACROBLOCKD *xd) {
  int i;
  for (i = 0; i < MAX_MB_PLANE; i++) {
    struct macroblockd_plane *const pd = &xd->plane[i];
    memset(pd->above_context, 0, sizeof(ENTROPY_CONTEXT) * pd->n4_w);
    memset(pd->left_context, 0, sizeof(ENTROPY_CONTEXT) * pd->n4_h);
  }
}

static MODE_INFO *set_offsets_recon(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                                    int mi_row, int mi_col, int bw, int bh,
                                    int bwl, int bhl) {
  const int offset = mi_row * cm->mi_stride + mi_col;
  const TileInfo *const tile = &xd->tile;
  xd->mi = cm->mi_grid_visible + offset;

  set_plane_n4(xd, bw, bh, bwl, bhl);

  set_skip_context(xd, mi_row, mi_col);

  // Distance of Mb to the various image edges. These are specified to 8th pel
  // as they are always compared to values that are in 1/8th pel units
  set_mi_row_col(xd, tile, mi_row, bh, mi_col, bw, cm->mi_rows, cm->mi_cols);

  vp9_setup_dst_planes(xd->plane, get_frame_new_buffer(cm), mi_row, mi_col);
  return xd->mi[0];
}

static MODE_INFO *set_offsets(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                              BLOCK_SIZE bsize, int mi_row, int mi_col, int bw,
                              int bh, int x_mis, int y_mis, int bwl, int bhl) {
  const int offset = mi_row * cm->mi_stride + mi_col;
  int x, y;
  const TileInfo *const tile = &xd->tile;

  xd->mi = cm->mi_grid_visible + offset;
  xd->mi[0] = &cm->mi[offset];
  // TODO(slavarnway): Generate sb_type based on bwl and bhl, instead of
  // passing bsize from decode_partition().
  xd->mi[0]->sb_type = bsize;
  for (y = 0; y < y_mis; ++y)
    for (x = !y; x < x_mis; ++x) {
      xd->mi[y * cm->mi_stride + x] = xd->mi[0];
    }

  set_plane_n4(xd, bw, bh, bwl, bhl);

  set_skip_context(xd, mi_row, mi_col);

  // Distance of Mb to the various image edges. These are specified to 8th pel
  // as they are always compared to values that are in 1/8th pel units
  set_mi_row_col(xd, tile, mi_row, bh, mi_col, bw, cm->mi_rows, cm->mi_cols);

  vp9_setup_dst_planes(xd->plane, get_frame_new_buffer(cm), mi_row, mi_col);
  return xd->mi[0];
}

static INLINE int predict_recon_inter(MACROBLOCKD *xd, MODE_INFO *mi,
                                      TileWorkerData *twd,
                                      predict_recon_func func) {
  int eobtotal = 0;
  int plane;
  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
    const int num_4x4_w = pd->n4_w;
    const int num_4x4_h = pd->n4_h;
    const int step = (1 << tx_size);
    int row, col;
    const int max_blocks_wide =
        num_4x4_w + (xd->mb_to_right_edge >= 0
                         ? 0
                         : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
    const int max_blocks_high =
        num_4x4_h + (xd->mb_to_bottom_edge >= 0
                         ? 0
                         : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

    xd->max_blocks_wide = xd->mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
    xd->max_blocks_high = xd->mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

    for (row = 0; row < max_blocks_high; row += step)
      for (col = 0; col < max_blocks_wide; col += step)
        eobtotal += func(twd, mi, plane, row, col, tx_size);
  }
  return eobtotal;
}

static INLINE void predict_recon_intra(MACROBLOCKD *xd, MODE_INFO *mi,
                                       TileWorkerData *twd,
                                       intra_recon_func func) {
  int plane;
  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
    const int num_4x4_w = pd->n4_w;
    const int num_4x4_h = pd->n4_h;
    const int step = (1 << tx_size);
    int row, col;
    const int max_blocks_wide =
        num_4x4_w + (xd->mb_to_right_edge >= 0
                         ? 0
                         : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
    const int max_blocks_high =
        num_4x4_h + (xd->mb_to_bottom_edge >= 0
                         ? 0
                         : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

    xd->max_blocks_wide = xd->mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
    xd->max_blocks_high = xd->mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

    for (row = 0; row < max_blocks_high; row += step)
      for (col = 0; col < max_blocks_wide; col += step)
        func(twd, mi, plane, row, col, tx_size);
  }
}

static void detoken_block(TileWorkerData *twd, MODE_INFO *mi, frameBuf *frameBuffer, int mi_col, int mi_row) {
  MACROBLOCKD *const xd = &twd->xd;
  int plane;
  if (!is_inter_block(mi)) {
    for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
      const struct macroblockd_plane *const pd = &xd->plane[plane];
      const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
      const int num_4x4_w = pd->n4_w;
      const int num_4x4_h = pd->n4_h;
      const int step = (1 << tx_size);
      int row, col;
      const int max_blocks_wide = num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
      const int max_blocks_high = num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

      PREDICTION_MODE mode = (plane == 0) ? mi->mode : mi->uv_mode;
      const int stride = pd->dst.stride;
      int n = 0;
      switch (tx_size) {
        case TX_4X4: n = 16; break;
        case TX_8X8: n = 64; break;
        case TX_16X16: n = 256; break;
        case TX_32X32: n = 1024; break;
        default: assert(0 && "Invalid transform size");
      }

      int by = (mi_row * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_y);
      int bx = (mi_col * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_x);

      int *eob_buf = frameBuffer->plane_eob[plane] + by * stride + bx;

      for (row = 0; row < max_blocks_high; row += step)
        for (col = 0; col < max_blocks_wide; col += step) {
          if (mi->sb_type < BLOCK_8X8)
            if (plane == 0) mode = xd->mi[0]->bmi[(row << 1) + col].as_mode;

          const TX_TYPE tx_type = (plane || xd->lossless) ? DCT_DCT : intra_mode_to_tx_type_lookup[mode];
          const scan_order *sc = (plane || xd->lossless) ? &vp9_default_scan_orders[tx_size] : &vp9_scan_orders[tx_size][tx_type];
          const int eob = vp9_decode_block_tokens(twd, plane, sc, col, row, tx_size, mi->segment_id);
          tran_low_t *dq = pd->dqcoeff;
          memcpy(frameBuffer->dqcoeff[plane], dq, n * sizeof(tran_low_t));

          if (eob > 0) {
            if (eob == 1) {
              dq[0] = 0;
            } else {
              if (tx_type == DCT_DCT && tx_size <= TX_16X16 && eob <= 10) memset(dq, 0, 4 * (4 << tx_size) * sizeof(dq[0]));
              else if (tx_size == TX_32X32 && eob <= 34) memset(dq, 0, 256 * sizeof(dq[0]));
              else memset(dq, 0, (16 << (tx_size << 1)) * sizeof(dq[0]));
            }
          }
          eob_buf[4 * row * stride + 4 * col] = eob;

          frameBuffer->dqcoeff[plane] += n;
        }
    }
  } else {
    for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
      const struct macroblockd_plane *const pd = &xd->plane[plane];
      const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
      const int num_4x4_w = pd->n4_w;
      const int num_4x4_h = pd->n4_h;
      const int step = (1 << tx_size);
      int row, col;
      const int max_blocks_wide = num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
      const int max_blocks_high = num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

      int n;
      const int stride = pd->dst.stride;
      switch (tx_size) {
        case TX_4X4: n = 16; break;
        case TX_8X8: n = 64; break;
        case TX_16X16: n = 256; break;
        case TX_32X32: n = 1024; break;
        default: assert(0 && "Invalid transform size");
      }

      int by = (mi_row * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_y);
      int bx = (mi_col * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_x);

      int *eob_buf = frameBuffer->plane_eob[plane] + by * stride + bx;

      for (row = 0; row < max_blocks_high; row += step)
        for (col = 0; col < max_blocks_wide; col += step) {
          const scan_order *sc = &vp9_default_scan_orders[tx_size];
          const int eob = vp9_decode_block_tokens(twd, plane, sc, col, row, tx_size, mi->segment_id);

          tran_low_t *dq = pd->dqcoeff;
          memcpy(frameBuffer->dqcoeff[plane], dq, n * sizeof(tran_low_t));

          if (eob > 0) {
            if (eob == 1) {
              dq[0] = 0;
            } else {
              if (tx_size <= TX_16X16 && eob <= 10) memset(dq, 0, 4 * (4 << tx_size) * sizeof(dq[0]));
              else if (tx_size == TX_32X32 && eob <= 34) memset(dq, 0, 256 * sizeof(dq[0]));
              else memset(dq, 0, (16 << (tx_size << 1)) * sizeof(dq[0]));
            }
          }
          
          eob_buf[4 * row * stride + 4 * col] = eob;

          frameBuffer->dqcoeff[plane] += n;
        }
    }
  }
}

static void intra_decode(TileWorkerData *twd, MODE_INFO *mi, frameBuf *frameBuffer, ModeInfoBuf* MiBuf) {
  MACROBLOCKD *const xd = &twd->xd;
  int plane;
  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
    const int num_4x4_w = pd->n4_w;
    const int num_4x4_h = pd->n4_h;
    const int step = (1 << tx_size);
    int row, col;
    const int max_blocks_wide = num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
    const int max_blocks_high = num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

    PREDICTION_MODE mode = (plane == 0) ? mi->mode : mi->uv_mode;
    const int stride = pd->dst.stride;
    int n = 0;
    switch (tx_size) {
      case TX_4X4: n = 16; break;
      case TX_8X8: n = 64; break;
      case TX_16X16: n = 256; break;
      case TX_32X32: n = 1024; break;
      default: assert(0 && "Invalid transform size");
    }

    int by = (*MiBuf->mi_row * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_y);
    int bx = (*MiBuf->mi_col * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_x);

    tran_high_t *residuals_buf = frameBuffer->plane_residuals[plane] + by * stride + bx;
    int *eob_buf = frameBuffer->plane_eob[plane] + by * stride + bx;

    for (row = 0; row < max_blocks_high; row += step)
      for (col = 0; col < max_blocks_wide; col += step) {
        if (mi->sb_type < BLOCK_8X8)
          if (plane == 0) mode = xd->mi[0]->bmi[(row << 1) + col].as_mode;
        
        const TX_TYPE tx_type = (plane || xd->lossless) ? DCT_DCT : intra_mode_to_tx_type_lookup[mode];
        const int eob = eob_buf[4 * row * stride + 4 * col];
        if (eob > 0) {
          inverse_transform_block_intra(xd, plane, tx_type, tx_size, &residuals_buf[4 * row * stride + 4 * col],
              stride, eob, frameBuffer->dqcoeff[plane]);
        }
        frameBuffer->dqcoeff[plane] += n;
      }
  }
  
}

static void intra_predict_and_reconstruct(TileWorkerData *twd, MODE_INFO *mi, frameBuf *frameBuffer, ModeInfoBuf* MiBuf) {
  MACROBLOCKD *const xd = &twd->xd;
  int plane;
  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
    const int num_4x4_w = pd->n4_w;
    const int num_4x4_h = pd->n4_h;
    const int step = (1 << tx_size);
    int row, col;
    const int max_blocks_wide = num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
    const int max_blocks_high = num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

    xd->max_blocks_wide = xd->mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
    xd->max_blocks_high = xd->mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

    PREDICTION_MODE mode = (plane == 0) ? mi->mode : mi->uv_mode;

    uint8_t *dst;
    const int stride = pd->dst.stride;
    int by = (*MiBuf->mi_row * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_y);
    int bx = (*MiBuf->mi_col * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_x);

    tran_high_t *residuals_buf = frameBuffer->plane_residuals[plane] + by * stride + bx;
    int *eob_buf = frameBuffer->plane_eob[plane] + by * stride + bx;
    
    for (row = 0; row < max_blocks_high; row += step)
      for (col = 0; col < max_blocks_wide; col += step) {
        dst = &pd->dst.buf[4 * row * stride + 4 * col];

        if (mi->sb_type < BLOCK_8X8)
          if (plane == 0) mode = xd->mi[0]->bmi[(row << 1) + col].as_mode;

        vp9_predict_intra_block(xd, pd->n4_wl, tx_size, mode, dst, stride, dst, stride, col, row, plane);

        if (!mi->skip) {
          if (eob_buf[4 * row * stride + 4 * col] > 0) {
            block_sum(xd, dst, &residuals_buf[4 * row * stride + 4 * col], tx_size, stride);
          }
        }
      }
  }
}

static void inter_residual_sum(TileWorkerData *twd, MODE_INFO *mi, frameBuf *frameBuffer, ModeInfoBuf* MiBuf) {
  MACROBLOCKD *const xd = &twd->xd;
  int plane;
  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
    const int num_4x4_w = pd->n4_w;
    const int num_4x4_h = pd->n4_h;
    const int step = (1 << tx_size);
    int row, col;
    const int max_blocks_wide = num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
    const int max_blocks_high = num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));

    uint8_t *dst;
    const int stride = pd->dst.stride;

    int by = (*MiBuf->mi_row * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_y);
    int bx = (*MiBuf->mi_col * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_x);

    tran_high_t *residuals_buf = frameBuffer->plane_residuals[plane] + by * stride + bx;
    int *eob_buf = frameBuffer->plane_eob[plane] + by * stride + bx;

    for (row = 0; row < max_blocks_high; row += step)
      for (col = 0; col < max_blocks_wide; col += step) {
        dst = &pd->dst.buf[4 * row * stride + 4 * col];

        if (eob_buf[4 * row * stride + 4 * col] > 0) {
          block_sum(xd, dst, &residuals_buf[4 * row * stride + 4 * col], tx_size, stride);
        }
      }
  }
}

static void inter_decode(TileWorkerData *twd, MODE_INFO *mi, frameBuf *frameBuffer, int less8x8, ModeInfoBuf* MiBuf) {
  MACROBLOCKD *const xd = &twd->xd;
  int eobtotal = 0;
  int plane;

  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
    const int num_4x4_w = pd->n4_w;
    const int num_4x4_h = pd->n4_h;
    const int step = (1 << tx_size);
    int row, col;
    const int max_blocks_wide = num_4x4_w + (xd->mb_to_right_edge >= 0 ? 0 : xd->mb_to_right_edge >> (5 + pd->subsampling_x));
    const int max_blocks_high = num_4x4_h + (xd->mb_to_bottom_edge >= 0 ? 0 : xd->mb_to_bottom_edge >> (5 + pd->subsampling_y));
    
    int n;
    const int stride = pd->dst.stride;
    switch (tx_size) {
      case TX_4X4: n = 16; break;
      case TX_8X8: n = 64; break;
      case TX_16X16: n = 256; break;
      case TX_32X32: n = 1024; break;
      default: assert(0 && "Invalid transform size");
    }

    int by = (*MiBuf->mi_row * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_y);
    int bx = (*MiBuf->mi_col * MI_SIZE * 8) >> (3 + xd->plane[plane].subsampling_x);

    tran_high_t *residuals_buf = frameBuffer->plane_residuals[plane] + by * stride + bx;
    int *eob_buf = frameBuffer->plane_eob[plane] + by * stride + bx;
    
    for (row = 0; row < max_blocks_high; row += step)
      for (col = 0; col < max_blocks_wide; col += step) {
        const int eob = eob_buf[4 * row * stride + 4 * col];

        if (eob > 0) {
          inverse_transform_block_inter(xd, plane, tx_size, &residuals_buf[4 * row * stride + 4 * col], stride,
              eob, frameBuffer->dqcoeff[plane]);
        }

        eobtotal += eob;
        frameBuffer->dqcoeff[plane] += n;
      }
  }
  
  if (!less8x8 && eobtotal == 0) mi->skip = 1;
}

static void decode_block(TileWorkerData *twd, VP9Decoder *const pbi, int mi_row,
                         int mi_col, BLOCK_SIZE bsize, int bwl, int bhl,
                         frameBuf* frameBuffer, ModeInfoBuf* MiBuf, int* size, int* subsize_array) {
  VP9_COMMON *const cm = &pbi->common;
  const int less8x8 = bsize < BLOCK_8X8;
  const int bw = 1 << (bwl - 1);
  const int bh = 1 << (bhl - 1);
  const int x_mis = VPXMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = VPXMIN(bh, cm->mi_rows - mi_row);
  vpx_reader *r = &twd->bit_reader;
  MACROBLOCKD *const xd = &twd->xd;

  MODE_INFO * mi = set_offsets(cm, xd, bsize, mi_row, mi_col, bw, bh, x_mis,
                              y_mis, bwl, bhl);

  if (bsize >= BLOCK_8X8 && (cm->subsampling_x || cm->subsampling_y)) {
    const BLOCK_SIZE uv_subsize =
        ss_size_lookup[bsize][cm->subsampling_x][cm->subsampling_y];
    if (uv_subsize == BLOCK_INVALID)
      vpx_internal_error(xd->error_info, VPX_CODEC_CORRUPT_FRAME,
                         "Invalid block size.");
  }

  vp9_read_mode_info(twd, pbi, mi_row, mi_col, x_mis, y_mis);

  if (mi->skip) {
    dec_reset_skip_context(xd);
  }

  if (!mi->skip) detoken_block(twd, mi, frameBuffer, mi_col, mi_row);

  subsize_array[*size] = less8x8;
  MiBuf->bhl[*size] = bhl;
  MiBuf->bwl[*size] = bwl;
  MiBuf->mi[*size] = mi;
  MiBuf->mi_col[*size] = mi_col;
  MiBuf->mi_row[*size] = mi_row;
  ++*size;
  
  xd->corrupted |= vpx_reader_has_error(r);
  
  if (cm->lf.filter_level) {
    vp9_build_mask(cm, mi, mi_row, mi_col, bw, bh);
  }
}

static void recon_block(TileWorkerData *twd, VP9Decoder *const pbi, int mi_row,
                        int mi_col, BLOCK_SIZE bsize, int bwl, int bhl) {
  VP9_COMMON *const cm = &pbi->common;
  const int bw = 1 << (bwl - 1);
  const int bh = 1 << (bhl - 1);
  MACROBLOCKD *const xd = &twd->xd;

  MODE_INFO *mi = set_offsets_recon(cm, xd, mi_row, mi_col, bw, bh, bwl, bhl);

  if (bsize >= BLOCK_8X8 && (cm->subsampling_x || cm->subsampling_y)) {
    const BLOCK_SIZE uv_subsize =
        ss_size_lookup[bsize][cm->subsampling_x][cm->subsampling_y];
    if (uv_subsize == BLOCK_INVALID)
      vpx_internal_error(xd->error_info, VPX_CODEC_CORRUPT_FRAME,
                         "Invalid block size.");
  }

  if (!is_inter_block(mi)) {
    predict_recon_intra(xd, mi, twd,
                        predict_and_reconstruct_intra_block_row_mt);
  } else {
    // Prediction
    //dec_build_inter_predictors_sb(0, 0, 0, 0, 0, 0);

    // Reconstruction
    if (!mi->skip) {
      predict_recon_inter(xd, mi, twd, reconstruct_inter_block_row_mt);
    }
  }

  vp9_build_mask(cm, mi, mi_row, mi_col, bw, bh);
}

static void parse_block(TileWorkerData *twd, VP9Decoder *const pbi, int mi_row,
                        int mi_col, BLOCK_SIZE bsize, int bwl, int bhl) {
  VP9_COMMON *const cm = &pbi->common;
  const int bw = 1 << (bwl - 1);
  const int bh = 1 << (bhl - 1);
  const int x_mis = VPXMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = VPXMIN(bh, cm->mi_rows - mi_row);
  vpx_reader *r = &twd->bit_reader;
  MACROBLOCKD *const xd = &twd->xd;

  MODE_INFO *mi = set_offsets(cm, xd, bsize, mi_row, mi_col, bw, bh, x_mis,
                              y_mis, bwl, bhl);

  if (bsize >= BLOCK_8X8 && (cm->subsampling_x || cm->subsampling_y)) {
    const BLOCK_SIZE uv_subsize =
        ss_size_lookup[bsize][cm->subsampling_x][cm->subsampling_y];
    if (uv_subsize == BLOCK_INVALID)
      vpx_internal_error(xd->error_info, VPX_CODEC_CORRUPT_FRAME,
                         "Invalid block size.");
  }

  vp9_read_mode_info(twd, pbi, mi_row, mi_col, x_mis, y_mis);

  if (mi->skip) {
    dec_reset_skip_context(xd);
  }

  if (!is_inter_block(mi)) {
    predict_recon_intra(xd, mi, twd, parse_intra_block_row_mt);
  } else {
    if (!mi->skip) {
      tran_low_t *dqcoeff[MAX_MB_PLANE];
      int *eob[MAX_MB_PLANE];
      int plane;
      int eobtotal;
      // Based on eobtotal and bsize, this may be mi->skip may be set to true
      // In that case dqcoeff and eob need to be backed up and restored as
      // recon_block will not increment these pointers for skip cases
      for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
        const struct macroblockd_plane *const pd = &xd->plane[plane];
        dqcoeff[plane] = pd->dqcoeff;
        eob[plane] = pd->eob;
      }
      eobtotal = predict_recon_inter(xd, mi, twd, parse_inter_block_row_mt);

      if (bsize >= BLOCK_8X8 && eobtotal == 0) {
        mi->skip = 1;  // skip loopfilter
        for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
          struct macroblockd_plane *pd = &xd->plane[plane];
          pd->dqcoeff = dqcoeff[plane];
          pd->eob = eob[plane];
        }
      }
    }
  }

  xd->corrupted |= vpx_reader_has_error(r);
}

static INLINE int dec_partition_plane_context(TileWorkerData *twd, int mi_row,
                                              int mi_col, int bsl) {
  const PARTITION_CONTEXT *above_ctx = twd->xd.above_seg_context + mi_col;
  const PARTITION_CONTEXT *left_ctx =
      twd->xd.left_seg_context + (mi_row & MI_MASK);
  int above = (*above_ctx >> bsl) & 1, left = (*left_ctx >> bsl) & 1;

  //  assert(bsl >= 0);

  return (left * 2 + above) + bsl * PARTITION_PLOFFSET;
}

static INLINE void dec_update_partition_context(TileWorkerData *twd, int mi_row,
                                                int mi_col, BLOCK_SIZE subsize,
                                                int bw) {
  PARTITION_CONTEXT *const above_ctx = twd->xd.above_seg_context + mi_col;
  PARTITION_CONTEXT *const left_ctx =
      twd->xd.left_seg_context + (mi_row & MI_MASK);

  // update the partition context at the end notes. set partition bits
  // of block sizes larger than the current one to be one, and partition
  // bits of smaller block sizes to be zero.
  memset(above_ctx, partition_context_lookup[subsize].above, bw);
  memset(left_ctx, partition_context_lookup[subsize].left, bw);
}

static PARTITION_TYPE read_partition(TileWorkerData *twd, int mi_row,
                                     int mi_col, int has_rows, int has_cols,
                                     int bsl) {
  const int ctx = dec_partition_plane_context(twd, mi_row, mi_col, bsl);
  const vpx_prob *const probs = twd->xd.partition_probs[ctx];
  FRAME_COUNTS *counts = twd->xd.counts;
  PARTITION_TYPE p;
  vpx_reader *r = &twd->bit_reader;

  if (has_rows && has_cols)
    p = (PARTITION_TYPE)vpx_read_tree(r, vp9_partition_tree, probs);
  else if (!has_rows && has_cols)
    p = vpx_read(r, probs[1]) ? PARTITION_SPLIT : PARTITION_HORZ;
  else if (has_rows && !has_cols)
    p = vpx_read(r, probs[2]) ? PARTITION_SPLIT : PARTITION_VERT;
  else
    p = PARTITION_SPLIT;

  if (counts) ++counts->partition[ctx][p];

  return p;
}

// TODO(slavarnway): eliminate bsize and subsize in future commits
static void decode_partition(TileWorkerData *twd, VP9Decoder *const pbi,
                             int mi_row, int mi_col, BLOCK_SIZE bsize,
                             int n4x4_l2, frameBuf* frameBuffer, ModeInfoBuf* MiBuf,
                             int* size, int* subsize_array) {
  VP9_COMMON *const cm = &pbi->common;
  const int n8x8_l2 = n4x4_l2 - 1;
  const int num_8x8_wh = 1 << n8x8_l2;
  const int hbs = num_8x8_wh >> 1;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;
  const int has_rows = (mi_row + hbs) < cm->mi_rows;
  const int has_cols = (mi_col + hbs) < cm->mi_cols;
  MACROBLOCKD *const xd = &twd->xd;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  partition = read_partition(twd, mi_row, mi_col, has_rows, has_cols, n8x8_l2);
  subsize = subsize_lookup[partition][bsize];  // get_subsize(bsize, partition);
  if (!hbs) {
    // calculate bmode block dimensions (log 2)
    xd->bmode_blocks_wl = 1 >> !!(partition & PARTITION_VERT);
    xd->bmode_blocks_hl = 1 >> !!(partition & PARTITION_HORZ);

    decode_block(twd, pbi, mi_row, mi_col, subsize, 1, 1, frameBuffer, MiBuf, size, subsize_array);
  } else {
    switch (partition) {
      case PARTITION_NONE: {
        decode_block(twd, pbi, mi_row, mi_col, subsize, n4x4_l2, n4x4_l2,
                     frameBuffer, MiBuf, size, subsize_array);
        break;
      }
      case PARTITION_HORZ:
      {
        decode_block(twd, pbi, mi_row, mi_col, subsize, n4x4_l2, n8x8_l2,
                     frameBuffer, MiBuf, size, subsize_array);
        if (has_rows) {
          decode_block(twd, pbi, mi_row + hbs, mi_col, subsize, n4x4_l2,
                       n8x8_l2, frameBuffer, MiBuf, size, subsize_array);                        
        }
        break;
      }
      case PARTITION_VERT: {
        decode_block(twd, pbi, mi_row, mi_col, subsize, n8x8_l2, n4x4_l2,
                     frameBuffer, MiBuf, size, subsize_array);
        if (has_cols) {
          decode_block(twd, pbi, mi_row, mi_col + hbs, subsize, n8x8_l2,
                       n4x4_l2, frameBuffer, MiBuf, size, subsize_array);
        }
        break;
      }
      case PARTITION_SPLIT:
        decode_partition(twd, pbi, mi_row, mi_col, subsize, n8x8_l2, frameBuffer, MiBuf, size, subsize_array);
        decode_partition(twd, pbi, mi_row, mi_col + hbs, subsize, n8x8_l2,
                         frameBuffer, MiBuf, size, subsize_array);
        decode_partition(twd, pbi, mi_row + hbs, mi_col, subsize, n8x8_l2,
                         frameBuffer, MiBuf, size, subsize_array);
        decode_partition(twd, pbi, mi_row + hbs, mi_col + hbs, subsize, n8x8_l2,
                         frameBuffer, MiBuf, size, subsize_array);
        break;
      default: assert(0 && "Invalid partition type");
    }
  }

  //update partition context
  if (bsize >= BLOCK_8X8 &&
      (bsize == BLOCK_8X8 || partition != PARTITION_SPLIT))
    dec_update_partition_context(twd, mi_row, mi_col, subsize, num_8x8_wh);
}

static void process_partition(TileWorkerData *twd, VP9Decoder *const pbi,
                              int mi_row, int mi_col, BLOCK_SIZE bsize,
                              int n4x4_l2, int parse_recon_flag,
                              process_block_fn_t process_block) {
  VP9_COMMON *const cm = &pbi->common;
  const int n8x8_l2 = n4x4_l2 - 1;
  const int num_8x8_wh = 1 << n8x8_l2;
  const int hbs = num_8x8_wh >> 1;
  PARTITION_TYPE partition;
  BLOCK_SIZE subsize;
  const int has_rows = (mi_row + hbs) < cm->mi_rows;
  const int has_cols = (mi_col + hbs) < cm->mi_cols;
  MACROBLOCKD *const xd = &twd->xd;

  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;

  if (parse_recon_flag & PARSE) {
    *xd->partition =
        read_partition(twd, mi_row, mi_col, has_rows, has_cols, n8x8_l2);
  }

  partition = *xd->partition;
  xd->partition++;

  subsize = get_subsize(bsize, partition);
  if (!hbs) {
    // calculate bmode block dimensions (log 2)
    xd->bmode_blocks_wl = 1 >> !!(partition & PARTITION_VERT);
    xd->bmode_blocks_hl = 1 >> !!(partition & PARTITION_HORZ);
    process_block(twd, pbi, mi_row, mi_col, subsize, 1, 1);
  } else {
    switch (partition) {
      case PARTITION_NONE:
        process_block(twd, pbi, mi_row, mi_col, subsize, n4x4_l2, n4x4_l2);
        break;
      case PARTITION_HORZ:
        process_block(twd, pbi, mi_row, mi_col, subsize, n4x4_l2, n8x8_l2);
        if (has_rows)
          process_block(twd, pbi, mi_row + hbs, mi_col, subsize, n4x4_l2,
                        n8x8_l2);
        break;
      case PARTITION_VERT:
        process_block(twd, pbi, mi_row, mi_col, subsize, n8x8_l2, n4x4_l2);
        if (has_cols)
          process_block(twd, pbi, mi_row, mi_col + hbs, subsize, n8x8_l2,
                        n4x4_l2);
        break;
      case PARTITION_SPLIT:
        process_partition(twd, pbi, mi_row, mi_col, subsize, n8x8_l2,
                          parse_recon_flag, process_block);
        process_partition(twd, pbi, mi_row, mi_col + hbs, subsize, n8x8_l2,
                          parse_recon_flag, process_block);
        process_partition(twd, pbi, mi_row + hbs, mi_col, subsize, n8x8_l2,
                          parse_recon_flag, process_block);
        process_partition(twd, pbi, mi_row + hbs, mi_col + hbs, subsize,
                          n8x8_l2, parse_recon_flag, process_block);
        break;
      default: assert(0 && "Invalid partition type");
    }
  }

  if (parse_recon_flag & PARSE) {
    // update partition context
    if ((bsize == BLOCK_8X8 || partition != PARTITION_SPLIT) &&
        bsize >= BLOCK_8X8)
      dec_update_partition_context(twd, mi_row, mi_col, subsize, num_8x8_wh);
  }
}

static void setup_token_decoder(const uint8_t *data, const uint8_t *data_end,
                                size_t read_size,
                                struct vpx_internal_error_info *error_info,
                                vpx_reader *r, vpx_decrypt_cb decrypt_cb,
                                void *decrypt_state) {
  // Validate the calculated partition length. If the buffer described by the
  // partition can't be fully read then throw an error.
  if (!read_is_valid(data, read_size, data_end))
    vpx_internal_error(error_info, VPX_CODEC_CORRUPT_FRAME,
                       "Truncated packet or corrupt tile length");

  if (vpx_reader_init(r, data, read_size, decrypt_cb, decrypt_state))
    vpx_internal_error(error_info, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate bool decoder %d", 1);
}

static void read_coef_probs_common(vp9_coeff_probs_model *coef_probs,
                                   vpx_reader *r) {
  int i, j, k, l, m;

  if (vpx_read_bit(r))
    for (i = 0; i < PLANE_TYPES; ++i)
      for (j = 0; j < REF_TYPES; ++j)
        for (k = 0; k < COEF_BANDS; ++k)
          for (l = 0; l < BAND_COEFF_CONTEXTS(k); ++l)
            for (m = 0; m < UNCONSTRAINED_NODES; ++m)
              vp9_diff_update_prob(r, &coef_probs[i][j][k][l][m]);
}

static void read_coef_probs(FRAME_CONTEXT *fc, TX_MODE tx_mode, vpx_reader *r) {
  const TX_SIZE max_tx_size = tx_mode_to_biggest_tx_size[tx_mode];
  TX_SIZE tx_size;
  for (tx_size = TX_4X4; tx_size <= max_tx_size; ++tx_size)
    read_coef_probs_common(fc->coef_probs[tx_size], r);
}

static void setup_segmentation(struct segmentation *seg,
                               struct vpx_read_bit_buffer *rb) {
  int i, j;

  seg->update_map = 0;
  seg->update_data = 0;

  seg->enabled = vpx_rb_read_bit(rb);
  if (!seg->enabled) return;

  // Segmentation map update
  seg->update_map = vpx_rb_read_bit(rb);
  if (seg->update_map) {
    for (i = 0; i < SEG_TREE_PROBS; i++)
      seg->tree_probs[i] =
          vpx_rb_read_bit(rb) ? vpx_rb_read_literal(rb, 8) : MAX_PROB;

    seg->temporal_update = vpx_rb_read_bit(rb);
    if (seg->temporal_update) {
      for (i = 0; i < PREDICTION_PROBS; i++)
        seg->pred_probs[i] =
            vpx_rb_read_bit(rb) ? vpx_rb_read_literal(rb, 8) : MAX_PROB;
    } else {
      for (i = 0; i < PREDICTION_PROBS; i++) seg->pred_probs[i] = MAX_PROB;
    }
  }

  // Segmentation data update
  seg->update_data = vpx_rb_read_bit(rb);
  if (seg->update_data) {
    seg->abs_delta = vpx_rb_read_bit(rb);

    vp9_clearall_segfeatures(seg);

    for (i = 0; i < MAX_SEGMENTS; i++) {
      for (j = 0; j < SEG_LVL_MAX; j++) {
        int data = 0;
        const int feature_enabled = vpx_rb_read_bit(rb);
        if (feature_enabled) {
          vp9_enable_segfeature(seg, i, j);
          data = decode_unsigned_max(rb, vp9_seg_feature_data_max(j));
          if (vp9_is_segfeature_signed(j))
            data = vpx_rb_read_bit(rb) ? -data : data;
        }
        vp9_set_segdata(seg, i, j, data);
      }
    }
  }
}

static void setup_loopfilter(struct loopfilter *lf,
                             struct vpx_read_bit_buffer *rb) {
  lf->filter_level = vpx_rb_read_literal(rb, 6);
  lf->sharpness_level = vpx_rb_read_literal(rb, 3);

  // Read in loop filter deltas applied at the MB level based on mode or ref
  // frame.
  lf->mode_ref_delta_update = 0;

  lf->mode_ref_delta_enabled = vpx_rb_read_bit(rb);
  if (lf->mode_ref_delta_enabled) {
    lf->mode_ref_delta_update = vpx_rb_read_bit(rb);
    if (lf->mode_ref_delta_update) {
      int i;

      for (i = 0; i < MAX_REF_LF_DELTAS; i++)
        if (vpx_rb_read_bit(rb))
          lf->ref_deltas[i] = vpx_rb_read_signed_literal(rb, 6);

      for (i = 0; i < MAX_MODE_LF_DELTAS; i++)
        if (vpx_rb_read_bit(rb))
          lf->mode_deltas[i] = vpx_rb_read_signed_literal(rb, 6);
    }
  }
}

static INLINE int read_delta_q(struct vpx_read_bit_buffer *rb) {
  return vpx_rb_read_bit(rb) ? vpx_rb_read_signed_literal(rb, 4) : 0;
}

static void setup_quantization(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                               struct vpx_read_bit_buffer *rb) {
  cm->base_qindex = vpx_rb_read_literal(rb, QINDEX_BITS);
  cm->y_dc_delta_q = read_delta_q(rb);
  cm->uv_dc_delta_q = read_delta_q(rb);
  cm->uv_ac_delta_q = read_delta_q(rb);
  cm->dequant_bit_depth = cm->bit_depth;
  xd->lossless = cm->base_qindex == 0 && cm->y_dc_delta_q == 0 &&
                 cm->uv_dc_delta_q == 0 && cm->uv_ac_delta_q == 0;

#if CONFIG_VP9_HIGHBITDEPTH
  xd->bd = (int)cm->bit_depth;
#endif
}

static void setup_segmentation_dequant(VP9_COMMON *const cm) {
  // Build y/uv dequant values based on segmentation.
  if (cm->seg.enabled) {
    int i;
    for (i = 0; i < MAX_SEGMENTS; ++i) {
      const int qindex = vp9_get_qindex(&cm->seg, i, cm->base_qindex);
      cm->y_dequant[i][0] =
          vp9_dc_quant(qindex, cm->y_dc_delta_q, cm->bit_depth);
      cm->y_dequant[i][1] = vp9_ac_quant(qindex, 0, cm->bit_depth);
      cm->uv_dequant[i][0] =
          vp9_dc_quant(qindex, cm->uv_dc_delta_q, cm->bit_depth);
      cm->uv_dequant[i][1] =
          vp9_ac_quant(qindex, cm->uv_ac_delta_q, cm->bit_depth);
    }
  } else {
    const int qindex = cm->base_qindex;
    // When segmentation is disabled, only the first value is used.  The
    // remaining are don't cares.
    cm->y_dequant[0][0] = vp9_dc_quant(qindex, cm->y_dc_delta_q, cm->bit_depth);
    cm->y_dequant[0][1] = vp9_ac_quant(qindex, 0, cm->bit_depth);
    cm->uv_dequant[0][0] =
        vp9_dc_quant(qindex, cm->uv_dc_delta_q, cm->bit_depth);
    cm->uv_dequant[0][1] =
        vp9_ac_quant(qindex, cm->uv_ac_delta_q, cm->bit_depth);
  }
}

static INTERP_FILTER read_interp_filter(struct vpx_read_bit_buffer *rb) {
  const INTERP_FILTER literal_to_filter[] = { EIGHTTAP_SMOOTH, EIGHTTAP,
                                              EIGHTTAP_SHARP, BILINEAR };
  return vpx_rb_read_bit(rb) ? SWITCHABLE
                             : literal_to_filter[vpx_rb_read_literal(rb, 2)];
}

static void setup_render_size(VP9_COMMON *cm, struct vpx_read_bit_buffer *rb) {
  cm->render_width = cm->width;
  cm->render_height = cm->height;
  if (vpx_rb_read_bit(rb))
    vp9_read_frame_size(rb, &cm->render_width, &cm->render_height);
}

static void resize_mv_buffer(VP9_COMMON *cm) {
  vpx_free(cm->cur_frame->mvs);
  cm->cur_frame->mi_rows = cm->mi_rows;
  cm->cur_frame->mi_cols = cm->mi_cols;
  CHECK_MEM_ERROR(cm, cm->cur_frame->mvs,
                  (MV_REF *)vpx_calloc(cm->mi_rows * cm->mi_cols,
                                       sizeof(*cm->cur_frame->mvs)));
}

static void resize_context_buffers(VP9_COMMON *cm, int width, int height) {
#if CONFIG_SIZE_LIMIT
  if (width > DECODE_WIDTH_LIMIT || height > DECODE_HEIGHT_LIMIT)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Dimensions of %dx%d beyond allowed size of %dx%d.",
                       width, height, DECODE_WIDTH_LIMIT, DECODE_HEIGHT_LIMIT);
#endif
  if (cm->width != width || cm->height != height) {
    const int new_mi_rows =
        ALIGN_POWER_OF_TWO(height, MI_SIZE_LOG2) >> MI_SIZE_LOG2;
    const int new_mi_cols =
        ALIGN_POWER_OF_TWO(width, MI_SIZE_LOG2) >> MI_SIZE_LOG2;

    // Allocations in vp9_alloc_context_buffers() depend on individual
    // dimensions as well as the overall size.
    if (new_mi_cols > cm->mi_cols || new_mi_rows > cm->mi_rows) {
      if (vp9_alloc_context_buffers(cm, width, height)) {
        // The cm->mi_* values have been cleared and any existing context
        // buffers have been freed. Clear cm->width and cm->height to be
        // consistent and to force a realloc next time.
        cm->width = 0;
        cm->height = 0;
        vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                           "Failed to allocate context buffers");
      }
    } else {
      vp9_set_mb_mi(cm, width, height);
    }
    vp9_init_context_buffers(cm);
    cm->width = width;
    cm->height = height;
  }
  if (cm->cur_frame->mvs == NULL || cm->mi_rows > cm->cur_frame->mi_rows ||
      cm->mi_cols > cm->cur_frame->mi_cols) {
    resize_mv_buffer(cm);
  }
}

static void setup_frame_size(VP9_COMMON *cm, struct vpx_read_bit_buffer *rb) {
  int width, height;
  BufferPool *const pool = cm->buffer_pool;
  vp9_read_frame_size(rb, &width, &height);
  resize_context_buffers(cm, width, height);
  setup_render_size(cm, rb);

  if (vpx_realloc_frame_buffer(
          get_frame_new_buffer(cm), cm->width, cm->height, cm->subsampling_x,
          cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
          cm->use_highbitdepth,
#endif
          VP9_DEC_BORDER_IN_PIXELS, cm->byte_alignment,
          &pool->frame_bufs[cm->new_fb_idx].raw_frame_buffer, pool->get_fb_cb,
          pool->cb_priv)) {
    vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate frame buffer");
  }

  pool->frame_bufs[cm->new_fb_idx].released = 0;
  pool->frame_bufs[cm->new_fb_idx].buf.subsampling_x = cm->subsampling_x;
  pool->frame_bufs[cm->new_fb_idx].buf.subsampling_y = cm->subsampling_y;
  pool->frame_bufs[cm->new_fb_idx].buf.bit_depth = (unsigned int)cm->bit_depth;
  pool->frame_bufs[cm->new_fb_idx].buf.color_space = cm->color_space;
  pool->frame_bufs[cm->new_fb_idx].buf.color_range = cm->color_range;
  pool->frame_bufs[cm->new_fb_idx].buf.render_width = cm->render_width;
  pool->frame_bufs[cm->new_fb_idx].buf.render_height = cm->render_height;
}

static INLINE int valid_ref_frame_img_fmt(vpx_bit_depth_t ref_bit_depth,
                                          int ref_xss, int ref_yss,
                                          vpx_bit_depth_t this_bit_depth,
                                          int this_xss, int this_yss) {
  return ref_bit_depth == this_bit_depth && ref_xss == this_xss &&
         ref_yss == this_yss;
}

static void setup_frame_size_with_refs(VP9_COMMON *cm,
                                       struct vpx_read_bit_buffer *rb) {
  int width, height;
  int found = 0, i;
  int has_valid_ref_frame = 0;
  BufferPool *const pool = cm->buffer_pool;
  for (i = 0; i < REFS_PER_FRAME; ++i) {
    if (vpx_rb_read_bit(rb)) {
      if (cm->frame_refs[i].idx != INVALID_IDX) {
        YV12_BUFFER_CONFIG *const buf = cm->frame_refs[i].buf;
        width = buf->y_crop_width;
        height = buf->y_crop_height;
        found = 1;
        break;
      } else {
        vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                           "Failed to decode frame size");
      }
    }
  }

  if (!found) vp9_read_frame_size(rb, &width, &height);

  if (width <= 0 || height <= 0)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Invalid frame size");

  // Check to make sure at least one of frames that this frame references
  // has valid dimensions.
  for (i = 0; i < REFS_PER_FRAME; ++i) {
    RefBuffer *const ref_frame = &cm->frame_refs[i];
    has_valid_ref_frame |=
        (ref_frame->idx != INVALID_IDX &&
         valid_ref_frame_size(ref_frame->buf->y_crop_width,
                              ref_frame->buf->y_crop_height, width, height));
  }
  if (!has_valid_ref_frame)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Referenced frame has invalid size");
  for (i = 0; i < REFS_PER_FRAME; ++i) {
    RefBuffer *const ref_frame = &cm->frame_refs[i];
    if (ref_frame->idx == INVALID_IDX ||
        !valid_ref_frame_img_fmt(ref_frame->buf->bit_depth,
                                 ref_frame->buf->subsampling_x,
                                 ref_frame->buf->subsampling_y, cm->bit_depth,
                                 cm->subsampling_x, cm->subsampling_y))
      vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                         "Referenced frame has incompatible color format");
  }

  resize_context_buffers(cm, width, height);
  setup_render_size(cm, rb);

  if (vpx_realloc_frame_buffer(
          get_frame_new_buffer(cm), cm->width, cm->height, cm->subsampling_x,
          cm->subsampling_y,
#if CONFIG_VP9_HIGHBITDEPTH
          cm->use_highbitdepth,
#endif
          VP9_DEC_BORDER_IN_PIXELS, cm->byte_alignment,
          &pool->frame_bufs[cm->new_fb_idx].raw_frame_buffer, pool->get_fb_cb,
          pool->cb_priv)) {
    vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate frame buffer");
  }

  pool->frame_bufs[cm->new_fb_idx].released = 0;
  pool->frame_bufs[cm->new_fb_idx].buf.subsampling_x = cm->subsampling_x;
  pool->frame_bufs[cm->new_fb_idx].buf.subsampling_y = cm->subsampling_y;
  pool->frame_bufs[cm->new_fb_idx].buf.bit_depth = (unsigned int)cm->bit_depth;
  pool->frame_bufs[cm->new_fb_idx].buf.color_space = cm->color_space;
  pool->frame_bufs[cm->new_fb_idx].buf.color_range = cm->color_range;
  pool->frame_bufs[cm->new_fb_idx].buf.render_width = cm->render_width;
  pool->frame_bufs[cm->new_fb_idx].buf.render_height = cm->render_height;
}

static void setup_tile_info(VP9_COMMON *cm, struct vpx_read_bit_buffer *rb) {
  int min_log2_tile_cols, max_log2_tile_cols, max_ones;
  vp9_get_tile_n_bits(cm->mi_cols, &min_log2_tile_cols, &max_log2_tile_cols);

  // columns
  max_ones = max_log2_tile_cols - min_log2_tile_cols;
  cm->log2_tile_cols = min_log2_tile_cols;
  while (max_ones-- && vpx_rb_read_bit(rb)) cm->log2_tile_cols++;

  if (cm->log2_tile_cols > 6)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Invalid number of tile columns");

  // rows
  cm->log2_tile_rows = vpx_rb_read_bit(rb);
  if (cm->log2_tile_rows) cm->log2_tile_rows += vpx_rb_read_bit(rb);
}

// Reads the next tile returning its size and adjusting '*data' accordingly
// based on 'is_last'.
static void get_tile_buffer(const uint8_t *const data_end, int is_last,
                            struct vpx_internal_error_info *error_info,
                            const uint8_t **data, vpx_decrypt_cb decrypt_cb,
                            void *decrypt_state, TileBuffer *buf) {
  size_t size;

  if (!is_last) {
    if (!read_is_valid(*data, 4, data_end))
      vpx_internal_error(error_info, VPX_CODEC_CORRUPT_FRAME,
                         "Truncated packet or corrupt tile length");

    if (decrypt_cb) {
      uint8_t be_data[4];
      decrypt_cb(decrypt_state, *data, be_data, 4);
      size = mem_get_be32(be_data);
    } else {
      size = mem_get_be32(*data);
    }
    *data += 4;

    if (size > (size_t)(data_end - *data))
      vpx_internal_error(error_info, VPX_CODEC_CORRUPT_FRAME,
                         "Truncated packet or corrupt tile size");
  } else {
    size = data_end - *data;
  }

  buf->data = *data;
  buf->size = size;

  *data += size;
}

static void get_tile_buffers(VP9Decoder *pbi, const uint8_t *data,
                             const uint8_t *data_end, int tile_cols,
                             int tile_rows,
                             TileBuffer (*tile_buffers)[1 << 6]) {
  int r, c;

  for (r = 0; r < tile_rows; ++r) {
    for (c = 0; c < tile_cols; ++c) {
      const int is_last = (r == tile_rows - 1) && (c == tile_cols - 1);
      TileBuffer *const buf = &tile_buffers[r][c];
      buf->col = c;
      get_tile_buffer(data_end, is_last, &pbi->common.error, &data,
                      pbi->decrypt_cb, pbi->decrypt_state, buf);
    }
  }
}

static void map_write(RowMTWorkerData *const row_mt_worker_data, int map_idx,
                      int sync_idx) {
#if CONFIG_MULTITHREAD
  pthread_mutex_lock(&row_mt_worker_data->recon_sync_mutex[sync_idx]);
  row_mt_worker_data->recon_map[map_idx] = 1;
  pthread_cond_signal(&row_mt_worker_data->recon_sync_cond[sync_idx]);
  pthread_mutex_unlock(&row_mt_worker_data->recon_sync_mutex[sync_idx]);
#else
  (void)row_mt_worker_data;
  (void)map_idx;
  (void)sync_idx;
#endif  // CONFIG_MULTITHREAD
}

static void map_read(RowMTWorkerData *const row_mt_worker_data, int map_idx,
                     int sync_idx) {
#if CONFIG_MULTITHREAD
  volatile int8_t *map = row_mt_worker_data->recon_map + map_idx;
  pthread_mutex_t *const mutex =
      &row_mt_worker_data->recon_sync_mutex[sync_idx];
  pthread_mutex_lock(mutex);
  while (!(*map)) {
    pthread_cond_wait(&row_mt_worker_data->recon_sync_cond[sync_idx], mutex);
  }
  pthread_mutex_unlock(mutex);
#else
  (void)row_mt_worker_data;
  (void)map_idx;
  (void)sync_idx;
#endif  // CONFIG_MULTITHREAD
}

static int lpf_map_write_check(VP9LfSync *lf_sync, int row, int num_tile_cols) {
  int return_val = 0;
#if CONFIG_MULTITHREAD
  int corrupted;
  pthread_mutex_lock(lf_sync->lf_mutex);
  corrupted = lf_sync->corrupted;
  pthread_mutex_unlock(lf_sync->lf_mutex);
  if (!corrupted) {
    pthread_mutex_lock(&lf_sync->recon_done_mutex[row]);
    lf_sync->num_tiles_done[row] += 1;
    if (num_tile_cols == lf_sync->num_tiles_done[row]) return_val = 1;
    pthread_mutex_unlock(&lf_sync->recon_done_mutex[row]);
  }
#else
  (void)lf_sync;
  (void)row;
  (void)num_tile_cols;
#endif
  return return_val;
}

static void vp9_tile_done(VP9Decoder *pbi) {
#if CONFIG_MULTITHREAD
  int terminate;
  RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
  const int all_parse_done = 1 << pbi->common.log2_tile_cols;
  pthread_mutex_lock(&row_mt_worker_data->recon_done_mutex);
  row_mt_worker_data->num_tiles_done++;
  terminate = all_parse_done == row_mt_worker_data->num_tiles_done;
  pthread_mutex_unlock(&row_mt_worker_data->recon_done_mutex);
  if (terminate) {
    vp9_jobq_terminate(&row_mt_worker_data->jobq);
  }
#else
  (void)pbi;
#endif
}

static void vp9_jobq_alloc(VP9Decoder *pbi) {
  VP9_COMMON *const cm = &pbi->common;
  RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
  const int aligned_rows = mi_cols_aligned_to_sb(cm->mi_rows);
  const int sb_rows = aligned_rows >> MI_BLOCK_SIZE_LOG2;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const size_t jobq_size = (tile_cols * sb_rows * 2 + sb_rows) * sizeof(Job);

  if (jobq_size > row_mt_worker_data->jobq_size) {
    vpx_free(row_mt_worker_data->jobq_buf);
    CHECK_MEM_ERROR(cm, row_mt_worker_data->jobq_buf, vpx_calloc(1, jobq_size));
    vp9_jobq_init(&row_mt_worker_data->jobq, row_mt_worker_data->jobq_buf,
                  jobq_size);
    row_mt_worker_data->jobq_size = jobq_size;
  }
}

static void recon_tile_row(TileWorkerData *tile_data, VP9Decoder *pbi,
                           int mi_row, int is_last_row, VP9LfSync *lf_sync,
                           int cur_tile_col) {
  VP9_COMMON *const cm = &pbi->common;
  RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int aligned_cols = mi_cols_aligned_to_sb(cm->mi_cols);
  const int sb_cols = aligned_cols >> MI_BLOCK_SIZE_LOG2;
  const int cur_sb_row = mi_row >> MI_BLOCK_SIZE_LOG2;
  int mi_col_start = tile_data->xd.tile.mi_col_start;
  int mi_col_end = tile_data->xd.tile.mi_col_end;
  int mi_col;

  vp9_zero(tile_data->xd.left_context);
  vp9_zero(tile_data->xd.left_seg_context);
  for (mi_col = mi_col_start; mi_col < mi_col_end; mi_col += MI_BLOCK_SIZE) {
    const int c = mi_col >> MI_BLOCK_SIZE_LOG2;
    int plane;
    const int sb_num = (cur_sb_row * (aligned_cols >> MI_BLOCK_SIZE_LOG2) + c);

    // Top Dependency
    if (cur_sb_row) {
      map_read(row_mt_worker_data, ((cur_sb_row - 1) * sb_cols) + c,
               ((cur_sb_row - 1) * tile_cols) + cur_tile_col);
    }

    for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
      tile_data->xd.plane[plane].eob =
          row_mt_worker_data->eob[plane] + (sb_num << EOBS_PER_SB_LOG2);
      tile_data->xd.plane[plane].dqcoeff =
          row_mt_worker_data->dqcoeff[plane] + (sb_num << DQCOEFFS_PER_SB_LOG2);
    }
    tile_data->xd.partition =
        row_mt_worker_data->partition + (sb_num * PARTITIONS_PER_SB);
    process_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4, RECON,
                      recon_block);
    if (cm->lf.filter_level && !cm->skip_loop_filter) {
      // Queue LPF_JOB
      int is_lpf_job_ready = 0;

      if (mi_col + MI_BLOCK_SIZE >= mi_col_end) {
        // Checks if this row has been decoded in all tiles
        is_lpf_job_ready = lpf_map_write_check(lf_sync, cur_sb_row, tile_cols);

        if (is_lpf_job_ready) {
          Job lpf_job;
          lpf_job.job_type = LPF_JOB;
          if (cur_sb_row > 0) {
            lpf_job.row_num = mi_row - MI_BLOCK_SIZE;
            vp9_jobq_queue(&row_mt_worker_data->jobq, &lpf_job,
                           sizeof(lpf_job));
          }
          if (is_last_row) {
            lpf_job.row_num = mi_row;
            vp9_jobq_queue(&row_mt_worker_data->jobq, &lpf_job,
                           sizeof(lpf_job));
          }
        }
      }
    }
    map_write(row_mt_worker_data, (cur_sb_row * sb_cols) + c,
              (cur_sb_row * tile_cols) + cur_tile_col);
  }
}

static void parse_tile_row(TileWorkerData *tile_data, VP9Decoder *pbi,
                           int mi_row, int cur_tile_col, uint8_t **data_end) {
  int mi_col;
  VP9_COMMON *const cm = &pbi->common;
  RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
  TileInfo *tile = &tile_data->xd.tile;
  TileBuffer *const buf = &pbi->tile_buffers[cur_tile_col];
  const int aligned_cols = mi_cols_aligned_to_sb(cm->mi_cols);

  vp9_zero(tile_data->dqcoeff);
  vp9_tile_init(tile, cm, 0, cur_tile_col);

  /* Update reader only at the beginning of each row in a tile */
  if (mi_row == 0) {
    setup_token_decoder(buf->data, *data_end, buf->size, &tile_data->error_info,
                        &tile_data->bit_reader, pbi->decrypt_cb,
                        pbi->decrypt_state);
  }
  vp9_init_macroblockd(cm, &tile_data->xd, tile_data->dqcoeff);
  tile_data->xd.error_info = &tile_data->error_info;

  vp9_zero(tile_data->xd.left_context);
  vp9_zero(tile_data->xd.left_seg_context);
  for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end;
       mi_col += MI_BLOCK_SIZE) {
    const int r = mi_row >> MI_BLOCK_SIZE_LOG2;
    const int c = mi_col >> MI_BLOCK_SIZE_LOG2;
    int plane;
    const int sb_num = (r * (aligned_cols >> MI_BLOCK_SIZE_LOG2) + c);
    for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
      tile_data->xd.plane[plane].eob =
          row_mt_worker_data->eob[plane] + (sb_num << EOBS_PER_SB_LOG2);
      tile_data->xd.plane[plane].dqcoeff =
          row_mt_worker_data->dqcoeff[plane] + (sb_num << DQCOEFFS_PER_SB_LOG2);
    }
    tile_data->xd.partition =
        row_mt_worker_data->partition + sb_num * PARTITIONS_PER_SB;
    process_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4, PARSE,
                      parse_block);
  }
}

static int row_decode_worker_hook(void *arg1, void *arg2) {
  ThreadData *const thread_data = (ThreadData *)arg1;
  uint8_t **data_end = (uint8_t **)arg2;
  VP9Decoder *const pbi = thread_data->pbi;
  VP9_COMMON *const cm = &pbi->common;
  RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
  const int aligned_cols = mi_cols_aligned_to_sb(cm->mi_cols);
  const int aligned_rows = mi_cols_aligned_to_sb(cm->mi_rows);
  const int sb_rows = aligned_rows >> MI_BLOCK_SIZE_LOG2;
  const int tile_cols = 1 << cm->log2_tile_cols;
  Job job;
  LFWorkerData *lf_data = thread_data->lf_data;
  VP9LfSync *lf_sync = thread_data->lf_sync;
  volatile int corrupted = 0;
  TileWorkerData *volatile tile_data_recon = NULL;

  while (!vp9_jobq_dequeue(&row_mt_worker_data->jobq, &job, sizeof(job), 1)) {
    int mi_col;
    const int mi_row = job.row_num;

    if (job.job_type == LPF_JOB) {
      lf_data->start = mi_row;
      lf_data->stop = lf_data->start + MI_BLOCK_SIZE;

      if (cm->lf.filter_level && !cm->skip_loop_filter &&
          mi_row < cm->mi_rows) {
        vp9_loopfilter_job(lf_data, lf_sync);
      }
    } else if (job.job_type == RECON_JOB) {
      const int cur_sb_row = mi_row >> MI_BLOCK_SIZE_LOG2;
      const int is_last_row = sb_rows - 1 == cur_sb_row;
      int mi_col_start, mi_col_end;
      if (!tile_data_recon)
        CHECK_MEM_ERROR(cm, tile_data_recon,
                        vpx_memalign(32, sizeof(TileWorkerData)));

      tile_data_recon->xd = pbi->mb;
      vp9_tile_init(&tile_data_recon->xd.tile, cm, 0, job.tile_col);
      vp9_init_macroblockd(cm, &tile_data_recon->xd, tile_data_recon->dqcoeff);
      mi_col_start = tile_data_recon->xd.tile.mi_col_start;
      mi_col_end = tile_data_recon->xd.tile.mi_col_end;

      if (setjmp(tile_data_recon->error_info.jmp)) {
        const int sb_cols = aligned_cols >> MI_BLOCK_SIZE_LOG2;
        tile_data_recon->error_info.setjmp = 0;
        corrupted = 1;
        for (mi_col = mi_col_start; mi_col < mi_col_end;
             mi_col += MI_BLOCK_SIZE) {
          const int c = mi_col >> MI_BLOCK_SIZE_LOG2;
          map_write(row_mt_worker_data, (cur_sb_row * sb_cols) + c,
                    (cur_sb_row * tile_cols) + job.tile_col);
        }
        if (is_last_row) {
          vp9_tile_done(pbi);
        }
        continue;
      }

      tile_data_recon->error_info.setjmp = 1;
      tile_data_recon->xd.error_info = &tile_data_recon->error_info;

      recon_tile_row(tile_data_recon, pbi, mi_row, is_last_row, lf_sync,
                     job.tile_col);

      if (corrupted)
        vpx_internal_error(&tile_data_recon->error_info,
                           VPX_CODEC_CORRUPT_FRAME,
                           "Failed to decode tile data");

      if (is_last_row) {
        vp9_tile_done(pbi);
      }
    } else if (job.job_type == PARSE_JOB) {
      TileWorkerData *const tile_data = &pbi->tile_worker_data[job.tile_col];

      if (setjmp(tile_data->error_info.jmp)) {
        tile_data->error_info.setjmp = 0;
        corrupted = 1;
        vp9_tile_done(pbi);
        continue;
      }

      tile_data->xd = pbi->mb;
      tile_data->xd.counts =
          cm->frame_parallel_decoding_mode ? 0 : &tile_data->counts;

      tile_data->error_info.setjmp = 1;

      parse_tile_row(tile_data, pbi, mi_row, job.tile_col, data_end);

      corrupted |= tile_data->xd.corrupted;
      if (corrupted)
        vpx_internal_error(&tile_data->error_info, VPX_CODEC_CORRUPT_FRAME,
                           "Failed to decode tile data");

      /* Queue in the recon_job for this row */
      {
        Job recon_job;
        recon_job.row_num = mi_row;
        recon_job.tile_col = job.tile_col;
        recon_job.job_type = RECON_JOB;
        vp9_jobq_queue(&row_mt_worker_data->jobq, &recon_job,
                       sizeof(recon_job));
      }

      /* Queue next parse job */
      if (mi_row + MI_BLOCK_SIZE < cm->mi_rows) {
        Job parse_job;
        parse_job.row_num = mi_row + MI_BLOCK_SIZE;
        parse_job.tile_col = job.tile_col;
        parse_job.job_type = PARSE_JOB;
        vp9_jobq_queue(&row_mt_worker_data->jobq, &parse_job,
                       sizeof(parse_job));
      }
    }
  }

  vpx_free(tile_data_recon);
  return !corrupted;
}

static void initBuf(frameBuf *buffer, int n, VP9_COMMON *cm) {
  YV12_BUFFER_CONFIG *src = &cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf;
  buffer->residuals = (tran_high_t *)malloc(src->frame_size * sizeof(tran_high_t));
  buffer->eob = (int *)malloc(src->frame_size * sizeof(int));
  memset(buffer->residuals, 0, src->frame_size * sizeof(tran_high_t));

  const int uv_border_h = src->border >> src->subsampling_y;
  const int uv_border_w = src->border >> src->subsampling_x;
  
  const int byte_alignment = cm->byte_alignment;
  const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
  const uint64_t yplane_size = (src->y_height + 2 * src->border) * (uint64_t)src->y_stride + byte_alignment;
  const uint64_t uvplane_size = (src->uv_height + 2 * uv_border_h) * (uint64_t)src->uv_stride + byte_alignment;

  buffer->plane_residuals[0] = (tran_high_t*)yv12_align_addr(buffer->residuals + (src->border * src->y_stride) + src->border, vp9_byte_align);
  buffer->plane_residuals[1] = (tran_high_t*)yv12_align_addr(buffer->residuals + yplane_size + (uv_border_h * src->uv_stride) + uv_border_w, vp9_byte_align);
  buffer->plane_residuals[2] = (tran_high_t*)yv12_align_addr(buffer->residuals + yplane_size + uvplane_size + (uv_border_h * src->uv_stride) + uv_border_w, vp9_byte_align);

  buffer->plane_eob[0] = (int*)yv12_align_addr(buffer->eob + (src->border * src->y_stride) + src->border, vp9_byte_align);
  buffer->plane_eob[1] = (int*)yv12_align_addr(buffer->eob + yplane_size + (uv_border_h * src->uv_stride) + uv_border_w, vp9_byte_align);
  buffer->plane_eob[2] = (int*)yv12_align_addr(buffer->eob + yplane_size + uvplane_size + (uv_border_h * src->uv_stride) + uv_border_w, vp9_byte_align);

  
  buffer->dqcoeff = (tran_low_t **)malloc(MAX_MB_PLANE * sizeof(tran_low_t *));

  for (int plane = 0; plane < MAX_MB_PLANE; ++plane) {
    buffer->dqcoeff[plane] = (tran_low_t *)malloc(n * sizeof(tran_low_t));
  }
}

static void cloneBuf(frameBuf *buffer, frameBuf *strtBuf) {
  strtBuf->residuals = buffer->residuals;
  strtBuf->eob = buffer->eob;

  for (int plane = 0; plane < MAX_MB_PLANE; ++plane) {
    strtBuf->dqcoeff[plane] = buffer->dqcoeff[plane];
  }
}

static void gotoPtrBuf(frameBuf *buffer, frameBuf *strtBuf) {
  for (int plane = 0; plane < MAX_MB_PLANE; ++plane) {
    buffer->dqcoeff[plane] = strtBuf->dqcoeff[plane];
  }
}

static void freeBuf(frameBuf *buffer) {
  for (int plane = 0; plane < MAX_MB_PLANE; ++plane) {
    free(buffer->dqcoeff[plane]);
  }

  free(buffer->residuals);
  free(buffer->eob);
  free(buffer->dqcoeff);

  free(buffer);
}

extern int wrap_cuda_inter_prediction(int n, double *gpu_copy, double *gpu_run, int *size_for_mb, ModeInfoBuf* MiBuf, 
    VP9_COMMON *cm, VP9Decoder *pbi, int tile_rows, int tile_cols, tran_high_t* residuals);
extern int wrap_cuda_intra_prediction(double *gpu_copy, double *gpu_run, int *size_for_mb, ModeInfoBuf* MiBuf, 
    VP9_COMMON *cm, VP9Decoder *pbi, int tile_rows, int tile_cols, frameBuf* frameBuffer);
static const uint8_t *decode_tiles(VP9Decoder *pbi, const uint8_t *data, const uint8_t *data_end) {
  VP9_COMMON *const cm = &pbi->common;
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  const int aligned_cols = mi_cols_aligned_to_sb(cm->mi_cols);
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  TileBuffer tile_buffers[4][1 << 6];
  int tile_row, tile_col;
  int mi_row, mi_col;
  TileWorkerData *tile_data = NULL;
  frameBuf *frameBuffer = (frameBuf *)malloc(sizeof(frameBuf));
  int n = cm->width * cm->height;
  static int fr = 0;
  initBuf(frameBuffer, n, cm);

  frameBuf *strtBuffer = (frameBuf *)malloc(sizeof(frameBuf));
  
  strtBuffer->dqcoeff = (tran_low_t **)malloc(MAX_MB_PLANE * sizeof(tran_low_t *));

  ModeInfoBuf MiBuf;
  MiBuf.bwl = (int *)malloc(n / 16 * sizeof(int));
  MiBuf.bhl = (int *)malloc(n / 16 * sizeof(int));
  MiBuf.mi_col = (int *)malloc(n / 16 * sizeof(int));
  MiBuf.mi_row = (int *)malloc(n / 16 * sizeof(int));
  MiBuf.mi = (MODE_INFO **)malloc(n / 16 * sizeof(MODE_INFO*));
  int *subsize_array = (int *)malloc(n / 16 * sizeof(int));
  int *size_for_mb = (int *)malloc(n / 2048 * sizeof(int));
  memset(size_for_mb, 0, n / 2048 * sizeof(int));

  //remember start
  cloneBuf(frameBuffer, strtBuffer);
  ModeInfoBuf strt_mi_buf;
  strt_mi_buf.bhl = MiBuf.bhl;
  strt_mi_buf.bwl = MiBuf.bwl;
  strt_mi_buf.mi_col = MiBuf.mi_col;
  strt_mi_buf.mi_row = MiBuf.mi_row;
  strt_mi_buf.mi = MiBuf.mi;
  int *strt_subsize_array = subsize_array;
  int *strt_size_for_mb = size_for_mb;

  if (cm->lf.filter_level && !cm->skip_loop_filter && pbi->lf_worker.data1 == NULL) {
    CHECK_MEM_ERROR(cm, pbi->lf_worker.data1, vpx_memalign(32, sizeof(LFWorkerData)));
    pbi->lf_worker.hook = vp9_loop_filter_worker;
    if (pbi->max_threads > 1 && !winterface->reset(&pbi->lf_worker)) {
      vpx_internal_error(&cm->error, VPX_CODEC_ERROR, "Loop filter thread creation failed");
    }
  }

  if (cm->lf.filter_level && !cm->skip_loop_filter) {
    LFWorkerData *const lf_data = (LFWorkerData *)pbi->lf_worker.data1;
    // Be sure to sync as we might be resuming after a failed frame decode.
    winterface->sync(&pbi->lf_worker);
    vp9_loop_filter_data_reset(lf_data, get_frame_new_buffer(cm), cm, pbi->mb.plane);
  }

  assert(tile_rows <= 4);
  assert(tile_cols <= (1 << 6));

  // Note: this memset assumes above_context[0], [1] and [2]
  // are allocated as part of the same buffer.
  memset(cm->above_context, 0, sizeof(*cm->above_context) * MAX_MB_PLANE * 2 * aligned_cols);

  memset(cm->above_seg_context, 0,sizeof(*cm->above_seg_context) * aligned_cols);

  vp9_reset_lfm(cm);

  get_tile_buffers(pbi, data, data_end, tile_cols, tile_rows, tile_buffers);
  
  // Load all tile information into tile_data.
  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      const TileBuffer *const buf = &tile_buffers[tile_row][tile_col];
      tile_data = pbi->tile_worker_data + tile_cols * tile_row + tile_col;
      tile_data->xd = pbi->mb;
      tile_data->xd.corrupted = 0;
      tile_data->xd.counts = cm->frame_parallel_decoding_mode ? NULL : &cm->counts;
      vp9_zero(tile_data->dqcoeff);
      vp9_tile_init(&tile_data->xd.tile, cm, tile_row, tile_col);
      setup_token_decoder(buf->data, data_end, buf->size, &cm->error, &tile_data->bit_reader, pbi->decrypt_cb, pbi->decrypt_state);
      vp9_init_macroblockd(cm, &tile_data->xd, tile_data->dqcoeff);
    }
  }

  printf("frame %d", fr);
  printf("\n");

  //entropy decoder
  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    TileInfo tile;
    vp9_tile_set_row(&tile, cm, tile_row);
    for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
      for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
        const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
        tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
        vp9_tile_set_col(&tile, cm, col);
        vp9_zero(tile_data->xd.left_context);
        vp9_zero(tile_data->xd.left_seg_context);
        for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
          if (pbi->row_mt == 1) {
            int plane;
            RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
            for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
              tile_data->xd.plane[plane].eob = row_mt_worker_data->eob[plane];
              tile_data->xd.plane[plane].dqcoeff = row_mt_worker_data->dqcoeff[plane];
            }
            tile_data->xd.partition = row_mt_worker_data->partition;
            process_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4, PARSE, parse_block);

            for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
              tile_data->xd.plane[plane].eob = row_mt_worker_data->eob[plane];
              tile_data->xd.plane[plane].dqcoeff = row_mt_worker_data->dqcoeff[plane];
            }
            tile_data->xd.partition = row_mt_worker_data->partition;
            process_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4, RECON, recon_block);
          } else {
            decode_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4, frameBuffer, &MiBuf, size_for_mb, subsize_array);
            MiBuf.mi_col += *size_for_mb;
            MiBuf.mi += *size_for_mb;
            MiBuf.mi_row += *size_for_mb;
            MiBuf.bhl += *size_for_mb;
            MiBuf.bwl += *size_for_mb;
            subsize_array += *size_for_mb;
            ++size_for_mb;
          }
        }
      }
    }
  }

  //go to start
  gotoPtrBuf(frameBuffer, strtBuffer);
  MiBuf.bhl = strt_mi_buf.bhl;
  MiBuf.bwl = strt_mi_buf.bwl;
  MiBuf.mi_row = strt_mi_buf.mi_row;
  MiBuf.mi_col = strt_mi_buf.mi_col;
  MiBuf.mi = strt_mi_buf.mi;
  subsize_array = strt_subsize_array;
  size_for_mb = strt_size_for_mb;
  
  //frame idct
  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    TileInfo tile;
    vp9_tile_set_row(&tile, cm, tile_row);
    for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
      for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
        const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
        tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
        vp9_tile_set_col(&tile, cm, col);
        vp9_zero(tile_data->xd.left_context);
        vp9_zero(tile_data->xd.left_seg_context);
        for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
          if (pbi->row_mt == 1) {
            //new code
          } else {
            for (int i = 0; i < *size_for_mb; ++i) {
              const int bh = 1 << (*MiBuf.bhl - 1);
              const int bw = 1 << (*MiBuf.bwl - 1);
              const int x_mis = VPXMIN(bw, cm->mi_cols - *MiBuf.mi_col);
              const int y_mis = VPXMIN(bh, cm->mi_rows - *MiBuf.mi_row);
              MACROBLOCKD *const xd = &tile_data->xd;

              set_offsets(cm, xd, MiBuf.mi[0]->sb_type, *MiBuf.mi_row,*MiBuf.mi_col, bw, bh, x_mis, y_mis, *MiBuf.bwl, *MiBuf.bhl);

              if (!MiBuf.mi[0]->skip) {
                if (!is_inter_block(MiBuf.mi[0])) {
                  intra_decode(tile_data, MiBuf.mi[0], frameBuffer, &MiBuf);
                } else {
                  inter_decode(tile_data, MiBuf.mi[0], frameBuffer, *subsize_array, &MiBuf);
                }
              }
              
              ++MiBuf.mi_col;
              ++MiBuf.mi;
              ++MiBuf.mi_row;
              ++MiBuf.bhl;
              ++MiBuf.bwl;
              ++subsize_array;
            }
            ++size_for_mb;
          }
        }
      }
    }
  }

  // frame idct
  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    TileInfo tile;
    vp9_tile_set_row(&tile, cm, tile_row);
    for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end;
         mi_row += MI_BLOCK_SIZE) {
      for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
        const int col =
            pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
        tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
        vp9_tile_set_col(&tile, cm, col);
        vp9_zero(tile_data->xd.left_context);
        vp9_zero(tile_data->xd.left_seg_context);
        for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end;
             mi_col += MI_BLOCK_SIZE) {
          for (int i = 0; i < *size_for_mb; ++i) {
            const int bh = 1 << (*MiBuf.bhl - 1);
            const int bw = 1 << (*MiBuf.bwl - 1);
            const int x_mis = VPXMIN(bw, cm->mi_cols - *MiBuf.mi_col);
            const int y_mis = VPXMIN(bh, cm->mi_rows - *MiBuf.mi_row);
            MACROBLOCKD *const xd = &tile_data->xd;

            set_offsets(cm, xd, MiBuf.mi[0]->sb_type, *MiBuf.mi_row,
                        *MiBuf.mi_col, bw, bh, x_mis, y_mis, *MiBuf.bwl,
                        *MiBuf.bhl);

            if (!MiBuf.mi[0]->skip) {
              if (!is_inter_block(MiBuf.mi[0])) {
                intra_decode(tile_data, MiBuf.mi[0], frameBuffer, &MiBuf);
              } else {
                inter_decode(tile_data, MiBuf.mi[0], frameBuffer,
                             *subsize_array, &MiBuf);
              }
            }

            ++MiBuf.mi_col;
            ++MiBuf.mi;
            ++MiBuf.mi_row;
            ++MiBuf.bhl;
            ++MiBuf.bwl;
            ++subsize_array;
          }
          ++size_for_mb;
        }
      }
    }
  }
  
  if (cm->frame_type == INTER_FRAME) {
    MiBuf.bhl = strt_mi_buf.bhl;
    MiBuf.bwl = strt_mi_buf.bwl;
    MiBuf.mi_row = strt_mi_buf.mi_row;
    MiBuf.mi_col = strt_mi_buf.mi_col;
    MiBuf.mi = strt_mi_buf.mi;
    size_for_mb = strt_size_for_mb;

    double gpu_copy, gpu_run;
    
    wrap_cuda_inter_prediction(n, &gpu_copy, &gpu_run, size_for_mb, &MiBuf, cm, pbi, tile_rows, tile_cols, frameBuffer->residuals);
    
    printf("gpu_copy inter %f \n", gpu_copy);
    printf("gpu_run  inter%f \n", gpu_run);
  }
  
  //go to start
  gotoPtrBuf(frameBuffer, strtBuffer);
  MiBuf.bhl = strt_mi_buf.bhl;
  MiBuf.bwl = strt_mi_buf.bwl;
  MiBuf.mi_row = strt_mi_buf.mi_row;
  MiBuf.mi_col = strt_mi_buf.mi_col;
  MiBuf.mi = strt_mi_buf.mi;
  subsize_array = strt_subsize_array;
  size_for_mb = strt_size_for_mb;
  
  double gpu_copy, gpu_run;
  
  wrap_cuda_intra_prediction(&gpu_copy, &gpu_run, size_for_mb, &MiBuf, cm, pbi, tile_rows, tile_cols, frameBuffer);
  
  printf("gpu_copy intra %f \n", gpu_copy);
  printf("gpu_run intra %f \n", gpu_run);

  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    TileInfo tile;
    vp9_tile_set_row(&tile, cm, tile_row);
    for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
      for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
        const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
        tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
        vp9_tile_set_col(&tile, cm, col);
        vp9_zero(tile_data->xd.left_context);
        vp9_zero(tile_data->xd.left_seg_context);
        pbi->mb.corrupted |= tile_data->xd.corrupted;
        if (pbi->mb.corrupted) vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME, "Failed to decode tile data");
      }
    }
  }

  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    TileInfo tile;
    vp9_tile_set_row(&tile, cm, tile_row);
    for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end;
         mi_row += MI_BLOCK_SIZE) {
      // Loopfilter one row.
      if (cm->lf.filter_level && !cm->skip_loop_filter) {
        const int lf_start = mi_row - MI_BLOCK_SIZE;
        LFWorkerData *const lf_data = (LFWorkerData *)pbi->lf_worker.data1;

        // delay the loopfilter by 1 macroblock row.
        if (lf_start < 0) continue;

        // decoding has completed: finish up the loop filter in this thread.
        if (mi_row + MI_BLOCK_SIZE >= cm->mi_rows) continue;

        winterface->sync(&pbi->lf_worker);
        lf_data->start = lf_start;
        lf_data->stop = mi_row;
        if (pbi->max_threads > 1) {
          winterface->launch(&pbi->lf_worker);
        } else {
          winterface->execute(&pbi->lf_worker);
        }
      }
    }
  }

  // Loopfilter remaining rows in the frame.
  if (cm->lf.filter_level && !cm->skip_loop_filter) {
    LFWorkerData *const lf_data = (LFWorkerData *)pbi->lf_worker.data1;
    winterface->sync(&pbi->lf_worker);
    lf_data->start = lf_data->stop;
    lf_data->stop = cm->mi_rows;
    winterface->execute(&pbi->lf_worker);
  }

  // Get last tile data.
  tile_data = pbi->tile_worker_data + tile_cols * tile_rows - 1;

  freeBuf(strtBuffer);
  free(frameBuffer);

  free(strt_mi_buf.mi_col);
  free(strt_mi_buf.mi_row);
  free(strt_mi_buf.mi);
  free(strt_mi_buf.bhl);
  free(strt_mi_buf.bwl);
  free(subsize_array);
  free(strt_size_for_mb);

  ++fr;

  return vpx_reader_find_end(&tile_data->bit_reader);
}

static void set_rows_after_error(VP9LfSync *lf_sync, int start_row, int mi_rows,
                                 int num_tiles_left, int total_num_tiles) {
  do {
    int mi_row;
    const int aligned_rows = mi_cols_aligned_to_sb(mi_rows);
    const int sb_rows = (aligned_rows >> MI_BLOCK_SIZE_LOG2);
    const int corrupted = 1;
    for (mi_row = start_row; mi_row < mi_rows; mi_row += MI_BLOCK_SIZE) {
      const int is_last_row = (sb_rows - 1 == mi_row >> MI_BLOCK_SIZE_LOG2);
      vp9_set_row(lf_sync, total_num_tiles, mi_row >> MI_BLOCK_SIZE_LOG2,
                  is_last_row, corrupted);
    }
    /* If there are multiple tiles, the second tile should start marking row
     * progress from row 0.
     */
    start_row = 0;
  } while (num_tiles_left--);
}

// On entry 'tile_data->data_end' points to the end of the input frame, on exit
// it is updated to reflect the bitreader position of the final tile column if
// present in the tile buffer group or NULL otherwise.
static int tile_worker_hook(void *arg1, void *arg2) {
  TileWorkerData *const tile_data = (TileWorkerData *)arg1;
  VP9Decoder *const pbi = (VP9Decoder *)arg2;

  TileInfo *volatile tile = &tile_data->xd.tile;
  const int final_col = (1 << pbi->common.log2_tile_cols) - 1;
  const uint8_t *volatile bit_reader_end = NULL;
  VP9_COMMON *cm = &pbi->common;

  LFWorkerData *lf_data = tile_data->lf_data;
  VP9LfSync *lf_sync = tile_data->lf_sync;

  volatile int mi_row = 0;
  volatile int n = tile_data->buf_start;
  tile_data->error_info.setjmp = 1;

  if (setjmp(tile_data->error_info.jmp)) {
    tile_data->error_info.setjmp = 0;
    tile_data->xd.corrupted = 1;
    tile_data->data_end = NULL;
    if (pbi->lpf_mt_opt && cm->lf.filter_level && !cm->skip_loop_filter) {
      const int num_tiles_left = tile_data->buf_end - n;
      const int mi_row_start = mi_row;
      set_rows_after_error(lf_sync, mi_row_start, cm->mi_rows, num_tiles_left,
                           1 << cm->log2_tile_cols);
    }
    return 0;
  }

  tile_data->xd.corrupted = 0;

  do {
    int mi_col;
    const TileBuffer *const buf = pbi->tile_buffers + n;

    /* Initialize to 0 is safe since we do not deal with streams that have
     * more than one row of tiles. (So tile->mi_row_start will be 0)
     */
    assert(cm->log2_tile_rows == 0);
    mi_row = 0;
    vp9_zero(tile_data->dqcoeff);
    vp9_tile_init(tile, &pbi->common, 0, buf->col);
    setup_token_decoder(buf->data, tile_data->data_end, buf->size,
                        &tile_data->error_info, &tile_data->bit_reader,
                        pbi->decrypt_cb, pbi->decrypt_state);
    vp9_init_macroblockd(&pbi->common, &tile_data->xd, tile_data->dqcoeff);
    // init resets xd.error_info
    tile_data->xd.error_info = &tile_data->error_info;

    for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end;
         mi_row += MI_BLOCK_SIZE) {
      vp9_zero(tile_data->xd.left_context);
      vp9_zero(tile_data->xd.left_seg_context);
      for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end;
           mi_col += MI_BLOCK_SIZE) {
        decode_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4, 0, 0, 0, 0);
      }
      if (pbi->lpf_mt_opt && cm->lf.filter_level && !cm->skip_loop_filter) {
        const int aligned_rows = mi_cols_aligned_to_sb(cm->mi_rows);
        const int sb_rows = (aligned_rows >> MI_BLOCK_SIZE_LOG2);
        const int is_last_row = (sb_rows - 1 == mi_row >> MI_BLOCK_SIZE_LOG2);
        vp9_set_row(lf_sync, 1 << cm->log2_tile_cols,
                    mi_row >> MI_BLOCK_SIZE_LOG2, is_last_row,
                    tile_data->xd.corrupted);
      }
    }

    if (buf->col == final_col) {
      bit_reader_end = vpx_reader_find_end(&tile_data->bit_reader);
    }
  } while (!tile_data->xd.corrupted && ++n <= tile_data->buf_end);

  if (pbi->lpf_mt_opt && n < tile_data->buf_end && cm->lf.filter_level &&
      !cm->skip_loop_filter) {
    /* This was not incremented in the tile loop, so increment before tiles left
     * calculation
     */
    ++n;
    set_rows_after_error(lf_sync, 0, cm->mi_rows, tile_data->buf_end - n,
                         1 << cm->log2_tile_cols);
  }

  if (pbi->lpf_mt_opt && !tile_data->xd.corrupted && cm->lf.filter_level &&
      !cm->skip_loop_filter) {
    vp9_loopfilter_rows(lf_data, lf_sync);
  }

  tile_data->data_end = bit_reader_end;
  return !tile_data->xd.corrupted;
}

// sorts in descending order
static int compare_tile_buffers(const void *a, const void *b) {
  const TileBuffer *const buf_a = (const TileBuffer *)a;
  const TileBuffer *const buf_b = (const TileBuffer *)b;
  return (buf_a->size < buf_b->size) - (buf_a->size > buf_b->size);
}

static INLINE void init_mt(VP9Decoder *pbi) {
  int n;
  VP9_COMMON *const cm = &pbi->common;
  VP9LfSync *lf_row_sync = &pbi->lf_row_sync;
  const int aligned_mi_cols = mi_cols_aligned_to_sb(cm->mi_cols);
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();

  if (pbi->num_tile_workers == 0) {
    const int num_threads = pbi->max_threads;
    CHECK_MEM_ERROR(cm, pbi->tile_workers,
                    vpx_malloc(num_threads * sizeof(*pbi->tile_workers)));
    for (n = 0; n < num_threads; ++n) {
      VPxWorker *const worker = &pbi->tile_workers[n];
      ++pbi->num_tile_workers;

      winterface->init(worker);
      if (n < num_threads - 1 && !winterface->reset(worker)) {
        vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                           "Tile decoder thread creation failed");
      }
    }
  }

  // Initialize LPF
  if ((pbi->lpf_mt_opt || pbi->row_mt) && cm->lf.filter_level &&
      !cm->skip_loop_filter) {
    vp9_lpf_mt_init(lf_row_sync, cm, cm->lf.filter_level,
                    pbi->num_tile_workers);
  }

  // Note: this memset assumes above_context[0], [1] and [2]
  // are allocated as part of the same buffer.
  memset(cm->above_context, 0,
         sizeof(*cm->above_context) * MAX_MB_PLANE * 2 * aligned_mi_cols);

  memset(cm->above_seg_context, 0,
         sizeof(*cm->above_seg_context) * aligned_mi_cols);

  vp9_reset_lfm(cm);
}

static const uint8_t *decode_tiles_row_wise_mt(VP9Decoder *pbi,
                                               const uint8_t *data,
                                               const uint8_t *data_end) {
  VP9_COMMON *const cm = &pbi->common;
  RowMTWorkerData *const row_mt_worker_data = pbi->row_mt_worker_data;
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  const int num_workers = pbi->max_threads;
  int i, n;
  int col;
  int corrupted = 0;
  const int sb_rows = mi_cols_aligned_to_sb(cm->mi_rows) >> MI_BLOCK_SIZE_LOG2;
  const int sb_cols = mi_cols_aligned_to_sb(cm->mi_cols) >> MI_BLOCK_SIZE_LOG2;
  VP9LfSync *lf_row_sync = &pbi->lf_row_sync;
  YV12_BUFFER_CONFIG *const new_fb = get_frame_new_buffer(cm);

  assert(tile_cols <= (1 << 6));
  assert(tile_rows == 1);
  (void)tile_rows;

  memset(row_mt_worker_data->recon_map, 0,
         sb_rows * sb_cols * sizeof(*row_mt_worker_data->recon_map));

  init_mt(pbi);

  // Reset tile decoding hook
  for (n = 0; n < num_workers; ++n) {
    VPxWorker *const worker = &pbi->tile_workers[n];
    ThreadData *const thread_data = &pbi->row_mt_worker_data->thread_data[n];
    winterface->sync(worker);

    if (cm->lf.filter_level && !cm->skip_loop_filter) {
      thread_data->lf_sync = lf_row_sync;
      thread_data->lf_data = &thread_data->lf_sync->lfdata[n];
      vp9_loop_filter_data_reset(thread_data->lf_data, new_fb, cm,
                                 pbi->mb.plane);
    }

    thread_data->pbi = pbi;

    worker->hook = row_decode_worker_hook;
    worker->data1 = thread_data;
    worker->data2 = (void *)&row_mt_worker_data->data_end;
  }

  for (col = 0; col < tile_cols; ++col) {
    TileWorkerData *const tile_data = &pbi->tile_worker_data[col];
    tile_data->xd = pbi->mb;
    tile_data->xd.counts =
        cm->frame_parallel_decoding_mode ? NULL : &tile_data->counts;
  }

  /* Reset the jobq to start of the jobq buffer */
  vp9_jobq_reset(&row_mt_worker_data->jobq);
  row_mt_worker_data->num_tiles_done = 0;
  row_mt_worker_data->data_end = NULL;

  // Load tile data into tile_buffers
  get_tile_buffers(pbi, data, data_end, tile_cols, tile_rows,
                   &pbi->tile_buffers);

  // Initialize thread frame counts.
  if (!cm->frame_parallel_decoding_mode) {
    for (col = 0; col < tile_cols; ++col) {
      TileWorkerData *const tile_data = &pbi->tile_worker_data[col];
      vp9_zero(tile_data->counts);
    }
  }

  // queue parse jobs for 0th row of every tile
  for (col = 0; col < tile_cols; ++col) {
    Job parse_job;
    parse_job.row_num = 0;
    parse_job.tile_col = col;
    parse_job.job_type = PARSE_JOB;
    vp9_jobq_queue(&row_mt_worker_data->jobq, &parse_job, sizeof(parse_job));
  }

  for (i = 0; i < num_workers; ++i) {
    VPxWorker *const worker = &pbi->tile_workers[i];
    worker->had_error = 0;
    if (i == num_workers - 1) {
      winterface->execute(worker);
    } else {
      winterface->launch(worker);
    }
  }

  for (; n > 0; --n) {
    VPxWorker *const worker = &pbi->tile_workers[n - 1];
    // TODO(jzern): The tile may have specific error data associated with
    // its vpx_internal_error_info which could be propagated to the main info
    // in cm. Additionally once the threads have been synced and an error is
    // detected, there's no point in continuing to decode tiles.
    corrupted |= !winterface->sync(worker);
  }

  pbi->mb.corrupted = corrupted;

  {
    /* Set data end */
    TileWorkerData *const tile_data = &pbi->tile_worker_data[tile_cols - 1];
    row_mt_worker_data->data_end = vpx_reader_find_end(&tile_data->bit_reader);
  }

  // Accumulate thread frame counts.
  if (!cm->frame_parallel_decoding_mode) {
    for (i = 0; i < tile_cols; ++i) {
      TileWorkerData *const tile_data = &pbi->tile_worker_data[i];
      vp9_accumulate_frame_counts(&cm->counts, &tile_data->counts, 1);
    }
  }

  return row_mt_worker_data->data_end;
}

static const uint8_t *decode_tiles_mt(VP9Decoder *pbi, const uint8_t *data,
                                      const uint8_t *data_end) {
  VP9_COMMON *const cm = &pbi->common;
  const VPxWorkerInterface *const winterface = vpx_get_worker_interface();
  const uint8_t *bit_reader_end = NULL;
  VP9LfSync *lf_row_sync = &pbi->lf_row_sync;
  YV12_BUFFER_CONFIG *const new_fb = get_frame_new_buffer(cm);
  const int tile_cols = 1 << cm->log2_tile_cols;
  const int tile_rows = 1 << cm->log2_tile_rows;
  const int num_workers = VPXMIN(pbi->max_threads, tile_cols);
  int n;

  assert(tile_cols <= (1 << 6));
  assert(tile_rows == 1);
  (void)tile_rows;

  init_mt(pbi);

  // Reset tile decoding hook
  for (n = 0; n < num_workers; ++n) {
    VPxWorker *const worker = &pbi->tile_workers[n];
    TileWorkerData *const tile_data =
        &pbi->tile_worker_data[n + pbi->total_tiles];
    winterface->sync(worker);

    if (pbi->lpf_mt_opt && cm->lf.filter_level && !cm->skip_loop_filter) {
      tile_data->lf_sync = lf_row_sync;
      tile_data->lf_data = &tile_data->lf_sync->lfdata[n];
      vp9_loop_filter_data_reset(tile_data->lf_data, new_fb, cm, pbi->mb.plane);
      tile_data->lf_data->y_only = 0;
    }

    tile_data->xd = pbi->mb;
    tile_data->xd.counts =
        cm->frame_parallel_decoding_mode ? NULL : &tile_data->counts;
    worker->hook = tile_worker_hook;
    worker->data1 = tile_data;
    worker->data2 = pbi;
  }

  // Load tile data into tile_buffers
  get_tile_buffers(pbi, data, data_end, tile_cols, tile_rows,
                   &pbi->tile_buffers);

  // Sort the buffers based on size in descending order.
  qsort(pbi->tile_buffers, tile_cols, sizeof(pbi->tile_buffers[0]),
        compare_tile_buffers);

  if (num_workers == tile_cols) {
    // Rearrange the tile buffers such that the largest, and
    // presumably the most difficult, tile will be decoded in the main thread.
    // This should help minimize the number of instances where the main thread
    // is waiting for a worker to complete.
    const TileBuffer largest = pbi->tile_buffers[0];
    memmove(pbi->tile_buffers, pbi->tile_buffers + 1,
            (tile_cols - 1) * sizeof(pbi->tile_buffers[0]));
    pbi->tile_buffers[tile_cols - 1] = largest;
  } else {
    int start = 0, end = tile_cols - 2;
    TileBuffer tmp;

    // Interleave the tiles to distribute the load between threads, assuming a
    // larger tile implies it is more difficult to decode.
    while (start < end) {
      tmp = pbi->tile_buffers[start];
      pbi->tile_buffers[start] = pbi->tile_buffers[end];
      pbi->tile_buffers[end] = tmp;
      start += 2;
      end -= 2;
    }
  }

  // Initialize thread frame counts.
  if (!cm->frame_parallel_decoding_mode) {
    for (n = 0; n < num_workers; ++n) {
      TileWorkerData *const tile_data =
          (TileWorkerData *)pbi->tile_workers[n].data1;
      vp9_zero(tile_data->counts);
    }
  }

  {
    const int base = tile_cols / num_workers;
    const int remain = tile_cols % num_workers;
    int buf_start = 0;

    for (n = 0; n < num_workers; ++n) {
      const int count = base + (remain + n) / num_workers;
      VPxWorker *const worker = &pbi->tile_workers[n];
      TileWorkerData *const tile_data = (TileWorkerData *)worker->data1;

      tile_data->buf_start = buf_start;
      tile_data->buf_end = buf_start + count - 1;
      tile_data->data_end = data_end;
      buf_start += count;

      worker->had_error = 0;
      if (n == num_workers - 1) {
        assert(tile_data->buf_end == tile_cols - 1);
        winterface->execute(worker);
      } else {
        winterface->launch(worker);
      }
    }

    for (; n > 0; --n) {
      VPxWorker *const worker = &pbi->tile_workers[n - 1];
      TileWorkerData *const tile_data = (TileWorkerData *)worker->data1;
      // TODO(jzern): The tile may have specific error data associated with
      // its vpx_internal_error_info which could be propagated to the main info
      // in cm. Additionally once the threads have been synced and an error is
      // detected, there's no point in continuing to decode tiles.
      pbi->mb.corrupted |= !winterface->sync(worker);
      if (!bit_reader_end) bit_reader_end = tile_data->data_end;
    }
  }

  // Accumulate thread frame counts.
  if (!cm->frame_parallel_decoding_mode) {
    for (n = 0; n < num_workers; ++n) {
      TileWorkerData *const tile_data =
          (TileWorkerData *)pbi->tile_workers[n].data1;
      vp9_accumulate_frame_counts(&cm->counts, &tile_data->counts, 1);
    }
  }

  assert(bit_reader_end || pbi->mb.corrupted);
  return bit_reader_end;
}

static void error_handler(void *data) {
  VP9_COMMON *const cm = (VP9_COMMON *)data;
  vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME, "Truncated packet");
}

static void read_bitdepth_colorspace_sampling(VP9_COMMON *cm,
                                              struct vpx_read_bit_buffer *rb) {
  if (cm->profile >= PROFILE_2) {
    cm->bit_depth = vpx_rb_read_bit(rb) ? VPX_BITS_12 : VPX_BITS_10;
#if CONFIG_VP9_HIGHBITDEPTH
    cm->use_highbitdepth = 1;
#endif
  } else {
    cm->bit_depth = VPX_BITS_8;
#if CONFIG_VP9_HIGHBITDEPTH
    cm->use_highbitdepth = 0;
#endif
  }
  cm->color_space = vpx_rb_read_literal(rb, 3);
  if (cm->color_space != VPX_CS_SRGB) {
    cm->color_range = (vpx_color_range_t)vpx_rb_read_bit(rb);
    if (cm->profile == PROFILE_1 || cm->profile == PROFILE_3) {
      cm->subsampling_x = vpx_rb_read_bit(rb);
      cm->subsampling_y = vpx_rb_read_bit(rb);
      if (cm->subsampling_x == 1 && cm->subsampling_y == 1)
        vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                           "4:2:0 color not supported in profile 1 or 3");
      if (vpx_rb_read_bit(rb))
        vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                           "Reserved bit set");
    } else {
      cm->subsampling_y = cm->subsampling_x = 1;
    }
  } else {
    cm->color_range = VPX_CR_FULL_RANGE;
    if (cm->profile == PROFILE_1 || cm->profile == PROFILE_3) {
      // Note if colorspace is SRGB then 4:4:4 chroma sampling is assumed.
      // 4:2:2 or 4:4:0 chroma sampling is not allowed.
      cm->subsampling_y = cm->subsampling_x = 0;
      if (vpx_rb_read_bit(rb))
        vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                           "Reserved bit set");
    } else {
      vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                         "4:4:4 color not supported in profile 0 or 2");
    }
  }
}

static INLINE void flush_all_fb_on_key(VP9_COMMON *cm) {
  if (cm->frame_type == KEY_FRAME && cm->current_video_frame > 0) {
    RefCntBuffer *const frame_bufs = cm->buffer_pool->frame_bufs;
    BufferPool *const pool = cm->buffer_pool;
    int i;
    for (i = 0; i < FRAME_BUFFERS; ++i) {
      if (i == cm->new_fb_idx) continue;
      frame_bufs[i].ref_count = 0;
      if (!frame_bufs[i].released) {
        pool->release_fb_cb(pool->cb_priv, &frame_bufs[i].raw_frame_buffer);
        frame_bufs[i].released = 1;
      }
    }
  }
}

static size_t read_uncompressed_header(VP9Decoder *pbi,
                                       struct vpx_read_bit_buffer *rb) {
  VP9_COMMON *const cm = &pbi->common;
  BufferPool *const pool = cm->buffer_pool;
  RefCntBuffer *const frame_bufs = pool->frame_bufs;
  int i, mask, ref_index = 0;
  size_t sz;

  cm->last_frame_type = cm->frame_type;
  cm->last_intra_only = cm->intra_only;

  if (vpx_rb_read_literal(rb, 2) != VP9_FRAME_MARKER)
    vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                       "Invalid frame marker");

  cm->profile = vp9_read_profile(rb);
#if CONFIG_VP9_HIGHBITDEPTH
  if (cm->profile >= MAX_PROFILES)
    vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                       "Unsupported bitstream profile");
#else
  if (cm->profile >= PROFILE_2)
    vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                       "Unsupported bitstream profile");
#endif

  cm->show_existing_frame = vpx_rb_read_bit(rb);
  if (cm->show_existing_frame) {
    // Show an existing frame directly.
    const int frame_to_show = cm->ref_frame_map[vpx_rb_read_literal(rb, 3)];
    if (frame_to_show < 0 || frame_bufs[frame_to_show].ref_count < 1) {
      vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                         "Buffer %d does not contain a decoded frame",
                         frame_to_show);
    }

    ref_cnt_fb(frame_bufs, &cm->new_fb_idx, frame_to_show);
    pbi->refresh_frame_flags = 0;
    cm->lf.filter_level = 0;
    cm->show_frame = 1;

    return 0;
  }

  cm->frame_type = (FRAME_TYPE)vpx_rb_read_bit(rb);
  cm->show_frame = vpx_rb_read_bit(rb);
  cm->error_resilient_mode = vpx_rb_read_bit(rb);

  if (cm->frame_type == KEY_FRAME) {
    if (!vp9_read_sync_code(rb))
      vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                         "Invalid frame sync code");

    read_bitdepth_colorspace_sampling(cm, rb);
    pbi->refresh_frame_flags = (1 << REF_FRAMES) - 1;

    for (i = 0; i < REFS_PER_FRAME; ++i) {
      cm->frame_refs[i].idx = INVALID_IDX;
      cm->frame_refs[i].buf = NULL;
    }

    setup_frame_size(cm, rb);
    if (pbi->need_resync) {
      memset(&cm->ref_frame_map, -1, sizeof(cm->ref_frame_map));
      flush_all_fb_on_key(cm);
      pbi->need_resync = 0;
    }
  } else {
    cm->intra_only = cm->show_frame ? 0 : vpx_rb_read_bit(rb);

    cm->reset_frame_context =
        cm->error_resilient_mode ? 0 : vpx_rb_read_literal(rb, 2);

    if (cm->intra_only) {
      if (!vp9_read_sync_code(rb))
        vpx_internal_error(&cm->error, VPX_CODEC_UNSUP_BITSTREAM,
                           "Invalid frame sync code");
      if (cm->profile > PROFILE_0) {
        read_bitdepth_colorspace_sampling(cm, rb);
      } else {
        // NOTE: The intra-only frame header does not include the specification
        // of either the color format or color sub-sampling in profile 0. VP9
        // specifies that the default color format should be YUV 4:2:0 in this
        // case (normative).
        cm->color_space = VPX_CS_BT_601;
        cm->color_range = VPX_CR_STUDIO_RANGE;
        cm->subsampling_y = cm->subsampling_x = 1;
        cm->bit_depth = VPX_BITS_8;
#if CONFIG_VP9_HIGHBITDEPTH
        cm->use_highbitdepth = 0;
#endif
      }

      pbi->refresh_frame_flags = vpx_rb_read_literal(rb, REF_FRAMES);
      setup_frame_size(cm, rb);
      if (pbi->need_resync) {
        memset(&cm->ref_frame_map, -1, sizeof(cm->ref_frame_map));
        pbi->need_resync = 0;
      }
    } else if (pbi->need_resync != 1) { /* Skip if need resync */
      pbi->refresh_frame_flags = vpx_rb_read_literal(rb, REF_FRAMES);
      for (i = 0; i < REFS_PER_FRAME; ++i) {
        const int ref = vpx_rb_read_literal(rb, REF_FRAMES_LOG2);
        const int idx = cm->ref_frame_map[ref];
        RefBuffer *const ref_frame = &cm->frame_refs[i];
        ref_frame->idx = idx;
        ref_frame->buf = &frame_bufs[idx].buf;
        cm->ref_frame_sign_bias[LAST_FRAME + i] = vpx_rb_read_bit(rb);
      }

      setup_frame_size_with_refs(cm, rb);

      cm->allow_high_precision_mv = vpx_rb_read_bit(rb);
      cm->interp_filter = read_interp_filter(rb);

      for (i = 0; i < REFS_PER_FRAME; ++i) {
        RefBuffer *const ref_buf = &cm->frame_refs[i];
#if CONFIG_VP9_HIGHBITDEPTH
        vp9_setup_scale_factors_for_frame(
            &ref_buf->sf, ref_buf->buf->y_crop_width,
            ref_buf->buf->y_crop_height, cm->width, cm->height,
            cm->use_highbitdepth);
#else
        vp9_setup_scale_factors_for_frame(
            &ref_buf->sf, ref_buf->buf->y_crop_width,
            ref_buf->buf->y_crop_height, cm->width, cm->height);
#endif
      }
    }
  }
#if CONFIG_VP9_HIGHBITDEPTH
  get_frame_new_buffer(cm)->bit_depth = cm->bit_depth;
#endif
  get_frame_new_buffer(cm)->color_space = cm->color_space;
  get_frame_new_buffer(cm)->color_range = cm->color_range;
  get_frame_new_buffer(cm)->render_width = cm->render_width;
  get_frame_new_buffer(cm)->render_height = cm->render_height;

  if (pbi->need_resync) {
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Keyframe / intra-only frame required to reset decoder"
                       " state");
  }

  if (!cm->error_resilient_mode) {
    cm->refresh_frame_context = vpx_rb_read_bit(rb);
    cm->frame_parallel_decoding_mode = vpx_rb_read_bit(rb);
    if (!cm->frame_parallel_decoding_mode) vp9_zero(cm->counts);
  } else {
    cm->refresh_frame_context = 0;
    cm->frame_parallel_decoding_mode = 1;
  }

  // This flag will be overridden by the call to vp9_setup_past_independence
  // below, forcing the use of context 0 for those frame types.
  cm->frame_context_idx = vpx_rb_read_literal(rb, FRAME_CONTEXTS_LOG2);

  // Generate next_ref_frame_map.
  for (mask = pbi->refresh_frame_flags; mask; mask >>= 1) {
    if (mask & 1) {
      cm->next_ref_frame_map[ref_index] = cm->new_fb_idx;
      ++frame_bufs[cm->new_fb_idx].ref_count;
    } else {
      cm->next_ref_frame_map[ref_index] = cm->ref_frame_map[ref_index];
    }
    // Current thread holds the reference frame.
    if (cm->ref_frame_map[ref_index] >= 0)
      ++frame_bufs[cm->ref_frame_map[ref_index]].ref_count;
    ++ref_index;
  }

  for (; ref_index < REF_FRAMES; ++ref_index) {
    cm->next_ref_frame_map[ref_index] = cm->ref_frame_map[ref_index];
    // Current thread holds the reference frame.
    if (cm->ref_frame_map[ref_index] >= 0)
      ++frame_bufs[cm->ref_frame_map[ref_index]].ref_count;
  }
  pbi->hold_ref_buf = 1;

  if (frame_is_intra_only(cm) || cm->error_resilient_mode)
    vp9_setup_past_independence(cm);

  setup_loopfilter(&cm->lf, rb);
  setup_quantization(cm, &pbi->mb, rb);
  setup_segmentation(&cm->seg, rb);
  setup_segmentation_dequant(cm);

  setup_tile_info(cm, rb);
  if (pbi->row_mt == 1) {
    int num_sbs = 1;
    const int aligned_rows = mi_cols_aligned_to_sb(cm->mi_rows);
    const int sb_rows = aligned_rows >> MI_BLOCK_SIZE_LOG2;
    const int num_jobs = sb_rows << cm->log2_tile_cols;

    if (pbi->row_mt_worker_data == NULL) {
      CHECK_MEM_ERROR(cm, pbi->row_mt_worker_data,
                      vpx_calloc(1, sizeof(*pbi->row_mt_worker_data)));
#if CONFIG_MULTITHREAD
      pthread_mutex_init(&pbi->row_mt_worker_data->recon_done_mutex, NULL);
#endif
    }

    if (pbi->max_threads > 1) {
      const int aligned_cols = mi_cols_aligned_to_sb(cm->mi_cols);
      const int sb_cols = aligned_cols >> MI_BLOCK_SIZE_LOG2;

      num_sbs = sb_cols * sb_rows;
    }

    if (num_sbs > pbi->row_mt_worker_data->num_sbs ||
        num_jobs > pbi->row_mt_worker_data->num_jobs) {
      vp9_dec_free_row_mt_mem(pbi->row_mt_worker_data);
      vp9_dec_alloc_row_mt_mem(pbi->row_mt_worker_data, cm, num_sbs,
                               pbi->max_threads, num_jobs);
    }
    vp9_jobq_alloc(pbi);
  }
  sz = vpx_rb_read_literal(rb, 16);

  if (sz == 0)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Invalid header size");

  return sz;
}

static int read_compressed_header(VP9Decoder *pbi, const uint8_t *data,
                                  size_t partition_size) {
  VP9_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &pbi->mb;
  FRAME_CONTEXT *const fc = cm->fc;
  vpx_reader r;
  int k;

  if (vpx_reader_init(&r, data, partition_size, pbi->decrypt_cb,
                      pbi->decrypt_state))
    vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                       "Failed to allocate bool decoder 0");

  cm->tx_mode = xd->lossless ? ONLY_4X4 : read_tx_mode(&r);
  if (cm->tx_mode == TX_MODE_SELECT) read_tx_mode_probs(&fc->tx_probs, &r);
  read_coef_probs(fc, cm->tx_mode, &r);

  for (k = 0; k < SKIP_CONTEXTS; ++k)
    vp9_diff_update_prob(&r, &fc->skip_probs[k]);

  if (!frame_is_intra_only(cm)) {
    nmv_context *const nmvc = &fc->nmvc;
    int i, j;

    read_inter_mode_probs(fc, &r);

    if (cm->interp_filter == SWITCHABLE) read_switchable_interp_probs(fc, &r);

    for (i = 0; i < INTRA_INTER_CONTEXTS; i++)
      vp9_diff_update_prob(&r, &fc->intra_inter_prob[i]);

    cm->reference_mode = read_frame_reference_mode(cm, &r);
    if (cm->reference_mode != SINGLE_REFERENCE)
      vp9_setup_compound_reference_mode(cm);
    read_frame_reference_mode_probs(cm, &r);

    for (j = 0; j < BLOCK_SIZE_GROUPS; j++)
      for (i = 0; i < INTRA_MODES - 1; ++i)
        vp9_diff_update_prob(&r, &fc->y_mode_prob[j][i]);

    for (j = 0; j < PARTITION_CONTEXTS; ++j)
      for (i = 0; i < PARTITION_TYPES - 1; ++i)
        vp9_diff_update_prob(&r, &fc->partition_prob[j][i]);

    read_mv_probs(nmvc, cm->allow_high_precision_mv, &r);
  }

  return vpx_reader_has_error(&r);
}

static struct vpx_read_bit_buffer *init_read_bit_buffer(
    VP9Decoder *pbi, struct vpx_read_bit_buffer *rb, const uint8_t *data,
    const uint8_t *data_end, uint8_t clear_data[MAX_VP9_HEADER_SIZE]) {
  rb->bit_offset = 0;
  rb->error_handler = error_handler;
  rb->error_handler_data = &pbi->common;
  if (pbi->decrypt_cb) {
    const int n = (int)VPXMIN(MAX_VP9_HEADER_SIZE, data_end - data);
    pbi->decrypt_cb(pbi->decrypt_state, data, clear_data, n);
    rb->bit_buffer = clear_data;
    rb->bit_buffer_end = clear_data + n;
  } else {
    rb->bit_buffer = data;
    rb->bit_buffer_end = data_end;
  }
  return rb;
}

//------------------------------------------------------------------------------

int vp9_read_sync_code(struct vpx_read_bit_buffer *const rb) {
  return vpx_rb_read_literal(rb, 8) == VP9_SYNC_CODE_0 &&
         vpx_rb_read_literal(rb, 8) == VP9_SYNC_CODE_1 &&
         vpx_rb_read_literal(rb, 8) == VP9_SYNC_CODE_2;
}

void vp9_read_frame_size(struct vpx_read_bit_buffer *rb, int *width,
                         int *height) {
  *width = vpx_rb_read_literal(rb, 16) + 1;
  *height = vpx_rb_read_literal(rb, 16) + 1;
}

BITSTREAM_PROFILE vp9_read_profile(struct vpx_read_bit_buffer *rb) {
  int profile = vpx_rb_read_bit(rb);
  profile |= vpx_rb_read_bit(rb) << 1;
  if (profile > 2) profile += vpx_rb_read_bit(rb);
  return (BITSTREAM_PROFILE)profile;
}

void X_Fuel(VP9Decoder* pbi) {
    YV12_BUFFER_CONFIG *src = &pbi->common.buffer_pool->frame_bufs[pbi->common.new_fb_idx].buf;

    const int uv_border_h = src->border >> src->subsampling_y;
    const int uv_border_w = src->border >> src->subsampling_x;

    const int byte_alignment = pbi->common.byte_alignment;
    const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
    const uint64_t yplane_size = (src->y_height + 2 * src->border) * (uint64_t)src->y_stride + byte_alignment;
    const uint64_t uvplane_size = (src->uv_height + 2 * uv_border_h) * (uint64_t)src->uv_stride + byte_alignment;
    
    uint8_t *alloc = src->buffer_alloc;

#if CONFIG_VP9_HIGHBITDEPTH
    alloc = CONVERT_TO_BYTEPTR(alloc);
#endif

    uint8_t *y_buf = (uint8_t *)yv12_align_addr(alloc + (src->border * src->y_stride) + src->border, vp9_byte_align);
    uint8_t *u_buf = (uint8_t *)yv12_align_addr(alloc + yplane_size + (uv_border_h * src->uv_stride) + uv_border_w, vp9_byte_align);
    uint8_t *v_buf = (uint8_t *)yv12_align_addr(alloc + yplane_size + uvplane_size + (uv_border_h * src->uv_stride) + uv_border_w, vp9_byte_align);

    uint16_t *y_buf16;
    uint16_t *u_buf16;
    uint16_t *v_buf16;

#if CONFIG_VP9_HIGHBITDEPTH
    y_buf16 = CONVERT_TO_SHORTPTR(y_buf);
    u_buf16 = CONVERT_TO_SHORTPTR(u_buf);
    v_buf16 = CONVERT_TO_SHORTPTR(v_buf);
#endif

    for (int i = 0; i < src->y_height; ++i) {
      vpx_memset16(y_buf16 - src->border, y_buf16[0], src->border);
      vpx_memset16(y_buf16 + src->y_width, y_buf16[src->y_width - 1],src->border);
      y_buf16 += src->y_stride;
    }

    for (int i = 0; i < src->uv_height; ++i) {
      vpx_memset16(u_buf16 - uv_border_w, u_buf16[0], uv_border_w);
      vpx_memset16(u_buf16 + src->uv_width, u_buf16[src->uv_width - 1], uv_border_w);
      vpx_memset16(v_buf16 - uv_border_w, v_buf16[0], uv_border_w);
      vpx_memset16(v_buf16 + src->uv_width, v_buf16[src->uv_width - 1], uv_border_w);
      u_buf16 += src->uv_stride;
      v_buf16 += src->uv_stride;
    }
}

void vp9_decode_frame(VP9Decoder *pbi, const uint8_t *data,
                      const uint8_t *data_end, const uint8_t **p_data_end) {
  VP9_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &pbi->mb;
  struct vpx_read_bit_buffer rb;
  int context_updated = 0;
  uint8_t clear_data[MAX_VP9_HEADER_SIZE];
  const size_t first_partition_size = read_uncompressed_header(
      pbi, init_read_bit_buffer(pbi, &rb, data, data_end, clear_data));
  const int tile_rows = 1 << cm->log2_tile_rows;
  const int tile_cols = 1 << cm->log2_tile_cols;
  YV12_BUFFER_CONFIG *const new_fb = get_frame_new_buffer(cm);
#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
  bitstream_queue_set_frame_read(cm->current_video_frame * 2 + cm->show_frame);
#endif
#if CONFIG_MISMATCH_DEBUG
  mismatch_move_frame_idx_r();
#endif
  xd->cur_buf = new_fb;

  if (!first_partition_size) {
    // showing a frame directly
    *p_data_end = data + (cm->profile <= PROFILE_2 ? 1 : 2);
    return;
  }

  data += vpx_rb_bytes_read(&rb);
  if (!read_is_valid(data, first_partition_size, data_end))
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Truncated packet or corrupt header length");

  cm->use_prev_frame_mvs =
      !cm->error_resilient_mode && cm->width == cm->last_width &&
      cm->height == cm->last_height && !cm->last_intra_only &&
      cm->last_show_frame && (cm->last_frame_type != KEY_FRAME);

  vp9_setup_block_planes(xd, cm->subsampling_x, cm->subsampling_y);

  *cm->fc = cm->frame_contexts[cm->frame_context_idx];
  if (!cm->fc->initialized)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Uninitialized entropy context.");

  xd->corrupted = 0;
  new_fb->corrupted = read_compressed_header(pbi, data, first_partition_size);
  if (new_fb->corrupted)
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Decode failed. Frame data header is corrupted.");

  if (cm->lf.filter_level && !cm->skip_loop_filter) {
    vp9_loop_filter_frame_init(cm, cm->lf.filter_level);
  }

  if (pbi->tile_worker_data == NULL ||
      (tile_cols * tile_rows) != pbi->total_tiles) {
    const int num_tile_workers =
        tile_cols * tile_rows + ((pbi->max_threads > 1) ? pbi->max_threads : 0);
    const size_t twd_size = num_tile_workers * sizeof(*pbi->tile_worker_data);
    // Ensure tile data offsets will be properly aligned. This may fail on
    // platforms without DECLARE_ALIGNED().
    assert((sizeof(*pbi->tile_worker_data) % 16) == 0);
    vpx_free(pbi->tile_worker_data);
    CHECK_MEM_ERROR(cm, pbi->tile_worker_data, vpx_memalign(32, twd_size));
    pbi->total_tiles = tile_rows * tile_cols;
  }

  if (pbi->max_threads > 1 && tile_rows == 1 &&
      (tile_cols > 1 || pbi->row_mt == 1)) {
    if (pbi->row_mt == 1) {
      *p_data_end =
          decode_tiles_row_wise_mt(pbi, data + first_partition_size, data_end);
    } else {
      // Multi-threaded tile decoder
      *p_data_end = decode_tiles_mt(pbi, data + first_partition_size, data_end);
      if (!pbi->lpf_mt_opt) {
        if (!xd->corrupted) {
          if (!cm->skip_loop_filter) {
            // If multiple threads are used to decode tiles, then we use those
            // threads to do parallel loopfiltering.
            vp9_loop_filter_frame_mt(
                new_fb, cm, pbi->mb.plane, cm->lf.filter_level, 0, 0,
                pbi->tile_workers, pbi->num_tile_workers, &pbi->lf_row_sync);
          }
        } else {
          vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                             "Decode failed. Frame data is corrupted.");
        }
      }
    }
  } else {
    *p_data_end = decode_tiles(pbi, data + first_partition_size, data_end);
    X_Fuel(pbi);
  }

  if (!xd->corrupted) {
    if (!cm->error_resilient_mode && !cm->frame_parallel_decoding_mode) {
      vp9_adapt_coef_probs(cm);

      if (!frame_is_intra_only(cm)) {
        vp9_adapt_mode_probs(cm);
        vp9_adapt_mv_probs(cm, cm->allow_high_precision_mv);
      }
    }
  } else {
    vpx_internal_error(&cm->error, VPX_CODEC_CORRUPT_FRAME,
                       "Decode failed. Frame data is corrupted.");
  }

  // Non frame parallel update frame context here.
  if (cm->refresh_frame_context && !context_updated)
    cm->frame_contexts[cm->frame_context_idx] = *cm->fc;
}
