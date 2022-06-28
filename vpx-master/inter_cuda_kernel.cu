
#include "inter_cuda_kernel.cuh"

#include "time.h"
#include <cstdio>
#include <cstring>

#include "assert.h"

__device__ static int64_t cuda_scaled_buffer_offset(int x_offset, int y_offset, int stride, const struct scale_factors* sf) {
	const int x = sf ? sf->scale_value_x(x_offset, sf) : x_offset;
	const int y = sf ? sf->scale_value_y(y_offset, sf) : y_offset;
	return (int64_t)y * stride + x;
}

__device__ static void cuda_setup_pred_plane(struct buf_2d* dst, uint8_t* src,
		int stride, int mi_row, int mi_col, const struct scale_factors* scale, int subsampling_x, int subsampling_y) {
	const int x = (MI_SIZE * mi_col) >> subsampling_x;
	const int y = (MI_SIZE * mi_row) >> subsampling_y;
	dst->buf = src + cuda_scaled_buffer_offset(x, y, stride, scale);
	dst->stride = stride;
}

__device__ void cuda_vp9_setup_pre_planes(MACROBLOCKD* xd, int idx, const YV12_BUFFER_CONFIG* src, int mi_row, int mi_col, const struct scale_factors* sf) {
	if (src != NULL) {
		int i;
		uint8_t* const buffers[MAX_MB_PLANE] = { src->y_buffer, src->u_buffer,
												 src->v_buffer };
		const int strides[MAX_MB_PLANE] = { src->y_stride, src->uv_stride,
											src->uv_stride };
		for (i = 0; i < MAX_MB_PLANE; ++i) {
			struct macroblockd_plane* const pd = &xd->plane[i];
			cuda_setup_pred_plane(&pd->pre[idx], buffers[i], strides[i], mi_row, mi_col,
				sf, pd->subsampling_x, pd->subsampling_y);
		}
	}
}

__device__ static int cuda_round_mv_comp_q2(int value) {
	return (value < 0 ? value - 1 : value + 1) / 2;
}

__device__ static int cuda_round_mv_comp_q4(int value) {
	return (value < 0 ? value - 2 : value + 2) / 4;
}

__device__ static MV cuda_mi_mv_pred_q2(const MODE_INFO* mi, int idx, int block0, int block1) {
	MV res = { cuda_round_mv_comp_q2(mi->bmi[block0].as_mv[idx].as_mv.row +
								mi->bmi[block1].as_mv[idx].as_mv.row),
			   cuda_round_mv_comp_q2(mi->bmi[block0].as_mv[idx].as_mv.col +
								mi->bmi[block1].as_mv[idx].as_mv.col) };
	return res;
}

__device__ static MV cuda_mi_mv_pred_q4(const MODE_INFO* mi, int idx) {
	MV res = { cuda_round_mv_comp_q4(mi->bmi[0].as_mv[idx].as_mv.row +
								mi->bmi[1].as_mv[idx].as_mv.row +
								mi->bmi[2].as_mv[idx].as_mv.row +
								mi->bmi[3].as_mv[idx].as_mv.row),
			   cuda_round_mv_comp_q4(mi->bmi[0].as_mv[idx].as_mv.col +
								mi->bmi[1].as_mv[idx].as_mv.col +
								mi->bmi[2].as_mv[idx].as_mv.col +
								mi->bmi[3].as_mv[idx].as_mv.col) };
	return res;
}

__device__ MV cuda_average_split_mvs(const struct macroblockd_plane* pd, const MODE_INFO* mi, int ref, int block) {
	const int ss_idx = ((pd->subsampling_x > 0) << 1) | (pd->subsampling_y > 0);
	MV res = { 0, 0 };
	switch (ss_idx) {
	case 0: res = mi->bmi[block].as_mv[ref].as_mv; break;
	case 1: res = cuda_mi_mv_pred_q2(mi, ref, block, block + 2); break;
	case 2: res = cuda_mi_mv_pred_q2(mi, ref, block, block + 1); break;
	case 3: res = cuda_mi_mv_pred_q4(mi, ref); break;
	default: assert(ss_idx <= 3 && ss_idx >= 0);
	}
	return res;
}

__device__ int cuda_clamp(int value, int low, int high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ static void cuda_clamp_mv(MV* mv, int min_col, int max_col, int min_row, int max_row) {
	mv->col = cuda_clamp(mv->col, min_col, max_col);
	mv->row = cuda_clamp(mv->row, min_row, max_row);
}

__device__ MV cuda_clamp_mv_to_umv_border_sb(const MACROBLOCKD* xd, const MV* src_mv, int bw, int bh, int ss_x, int ss_y) {
	const int spel_left = (VP9_INTERP_EXTEND + bw) << SUBPEL_BITS;
	const int spel_right = spel_left - SUBPEL_SHIFTS;
	const int spel_top = (VP9_INTERP_EXTEND + bh) << SUBPEL_BITS;
	const int spel_bottom = spel_top - SUBPEL_SHIFTS;
	MV clamped_mv = { (short)(src_mv->row * (1 << (1 - ss_y))),
					  (short)(src_mv->col * (1 << (1 - ss_x))) };
	assert(ss_x <= 1);
	assert(ss_y <= 1);

	cuda_clamp_mv(&clamped_mv, xd->mb_to_left_edge * (1 << (1 - ss_x)) - spel_left,
		xd->mb_to_right_edge * (1 << (1 - ss_x)) + spel_right,
		xd->mb_to_top_edge * (1 << (1 - ss_y)) - spel_top,
		xd->mb_to_bottom_edge * (1 << (1 - ss_y)) + spel_bottom);

	return clamped_mv;
}

__device__ int cuda_vp9_is_valid_scale(const struct scale_factors* sf)
{
	return sf->x_scale_fp != REF_INVALID_SCALE &&
		sf->y_scale_fp != REF_INVALID_SCALE;
}

__device__ int cuda_vp9_is_scaled(const struct scale_factors* sf)
{
	return cuda_vp9_is_valid_scale(sf) &&
		(sf->x_scale_fp != REF_NO_SCALE || sf->y_scale_fp != REF_NO_SCALE);
}

__device__ int unscaled_value(int val, const struct scale_factors* sf) {
	(void)sf;
	return val;
}

__device__ int scaled_x(int val, const struct scale_factors* sf)
{
	return (int)((int64_t)val * sf->x_scale_fp >> REF_SCALE_SHIFT);
}

__device__  int scaled_y(int val, const struct scale_factors* sf)
{
	return (int)((int64_t)val * sf->y_scale_fp >> REF_SCALE_SHIFT);
}

__device__ int cuda_valid_ref_frame_size(int ref_width, int ref_height, int this_width, int this_height)
{
	return 2 * this_width >= ref_width && 2 * this_height >= ref_height &&
		this_width <= 16 * ref_width && this_height <= 16 * ref_height;
}

__device__ int get_fixed_point_scale_factor(int other_size, int this_size)
{
	return (other_size << REF_SCALE_SHIFT) / this_size;
}

__device__ MV32 cuda_vp9_scale_mv(const MV* mv, int x, int y, const struct scale_factors* sf) {
	const int x_off_q4 = scaled_x(x << SUBPEL_BITS, sf) & SUBPEL_MASK;
	const int y_off_q4 = scaled_y(y << SUBPEL_BITS, sf) & SUBPEL_MASK;
	const MV32 res = { scaled_y(mv->row, sf) + y_off_q4,
					   scaled_x(mv->col, sf) + x_off_q4 };
	return res;
}

__device__ uint8_t cuda_clip_pixel(int val)
{
	return (val > 255) ? 255 : (val < 0) ? 0 : val;
}

__device__ double cuda_fclamp(double value, double low, double high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ int64_t cuda_lclamp(int64_t value, int64_t low, int64_t high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ uint16_t cuda_clip_pixel_highbd(int val, int bd)
{
	switch (bd)
	{
		case 8:
		default: return (uint16_t)cuda_clamp(val, 0, 255);
		case 10: return (uint16_t)cuda_clamp(val, 0, 1023);
		case 12: return (uint16_t)cuda_clamp(val, 0, 4095);
	}
}

__device__ static tran_high_t cuda_highbd_check_range(tran_high_t input, int bd) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
	// For valid highbitdepth VP9 streams, intermediate stage coefficients will
	// stay within the ranges:
	// - 8 bit: signed 16 bit integer
	// - 10 bit: signed 18 bit integer
	// - 12 bit: signed 20 bit integer
	const int32_t int_max = (1 << (7 + bd)) - 1;
	const int32_t int_min = -int_max - 1;
	assert(int_min <= input);
	assert(input <= int_max);
	(void)int_min;
#endif  // CONFIG_COEFFICIENT_RANGE_CHECKING
	(void)bd;
	return input;
}

__device__ static uint16_t cuda_highbd_clip_pixel_add(int dest, tran_high_t trans, int bd) {
	trans = cuda_highbd_check_range(trans, bd);
	return cuda_clip_pixel_highbd(dest + (int)trans, bd);
}

__device__ static void cuda_convolve_horiz(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                           const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint8_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_convolve_avg_horiz(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                               const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint8_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = ROUND_POWER_OF_TWO(
				dst[x] + cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)), 1);
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_convolve_vert(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                          const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint8_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k)
				sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ static void cuda_convolve_avg_vert(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                              const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint8_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k)
				sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = ROUND_POWER_OF_TWO(
				dst[y * dst_stride] +
				cuda_clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)),
				1);
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ void cuda_vpx_convolve8_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                           const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)y0_q4;
	(void)y_step_q4;
	cuda_convolve_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_avg_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                               const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)y0_q4;
	(void)y_step_q4;
	cuda_convolve_avg_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                          const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)x0_q4;
	(void)x_step_q4;
	cuda_convolve_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_avg_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                              const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	(void)x0_q4;
	(void)x_step_q4;
	cuda_convolve_avg_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_convolve8_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                     int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	uint8_t temp[64 * 135];
	const int intermediate_height =
		(((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

	assert(w <= 64);
	assert(h <= 64);
	assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
	assert(x_step_q4 <= 64);

	cuda_convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64, filter, x0_q4, x_step_q4, w, intermediate_height);
	cuda_convolve_vert(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride, filter, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_convolve_avg_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                        int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	int x, y;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;

	for (y = 0; y < h; ++y)
	{
		for (x = 0; x < w; ++x) dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_convolve8_avg_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	// Fixed size intermediate buffer places limits on parameters.
	DECLARE_ALIGNED(16, uint8_t, temp[64 * 64]);
	assert(w <= 64);
	assert(h <= 64);

	cuda_vpx_convolve8_c(src, src_stride, temp, 64, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
	cuda_vpx_convolve_avg_c(temp, 64, dst, dst_stride, NULL, 0, 0, 0, 0, w, h);
}

__device__ void cuda_vpx_convolve_copy_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {
	int r;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;

	for (r = h; r > 0; --r)
	{
		memcpy(dst, src, w);
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_scaled_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                        int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                       int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_2d_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                     int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_avg_horiz_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                            const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_avg_vert_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride,
                                           const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ void cuda_vpx_scaled_avg_2d_c(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter,
                                         int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h)
{
	cuda_vpx_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

__device__ static void cuda_highbd_convolve_horiz(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                  const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h, int bd) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	int b = 0;
	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint16_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[b + x] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
			x_q4 += x_step_q4;
		}
		src += src_stride;
		b += dst_stride;
	}
}

__device__ static void cuda_highbd_convolve_avg_horiz(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                      const InterpKernel* x_filters, int x0_q4, int x_step_q4, int w, int h, int bd) {
	int x, y;
	src -= SUBPEL_TAPS / 2 - 1;

	for (y = 0; y < h; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint16_t* const src_x = &src[x_q4 >> SUBPEL_BITS];
			const int16_t* const x_filter = x_filters[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			dst[x] = ROUND_POWER_OF_TWO(dst[x] + cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd), 1);
			x_q4 += x_step_q4;
		}
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_highbd_convolve_vert(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                 const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h, int bd) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint16_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ static void cuda_highbd_convolve_avg_vert(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                     const InterpKernel* y_filters, int y0_q4, int y_step_q4, int w, int h, int bd) {
	int x, y;
	src -= src_stride * (SUBPEL_TAPS / 2 - 1);

	for (x = 0; x < w; ++x)
	{
		int y_q4 = y0_q4;
		for (y = 0; y < h; ++y)
		{
			const uint16_t* src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
			const int16_t* const y_filter = y_filters[y_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k)
				sum += src_y[k * src_stride] * y_filter[k];
			dst[y * dst_stride] = ROUND_POWER_OF_TWO( dst[y * dst_stride] + cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd), 1);
			y_q4 += y_step_q4;
		}
		++src;
		++dst;
	}
}

__device__ static void shared_convolve_row(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
	const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int bd, int y0, int frame_h, int x0) {
	int sm[16];
	memset(sm, 0, 16 * sizeof(int));

	int y;
	const uint16_t* ref_row = src;
	if (y0 >= frame_h) ref_row += (frame_h - 1) * src_stride;
	else if (y0 > 0) ref_row += y0 * src_stride;

	int st = x0;
	x0 = x0 & ~1;
	int delta = st - x0;
	ref_row += x0;
	
	__shared__ int int_cashe[6 * interThreadsPerBlock];
	
	for (y = 0; y < 11; ++y)
	{
		for(int i = 0; i < 6; ++i) int_cashe[6 * threadIdx.x + i] = ((const int*)ref_row)[i];
		uint16_t *cashe = (uint16_t*)(int_cashe + 6 * threadIdx.x) + delta;
		
		const int16_t* x_filter = filter[x0_q4 & SUBPEL_MASK];
		int k, sum = 0;
		for (k = 0; k < SUBPEL_TAPS; ++k) sum += cashe[k] * x_filter[k];
		int a = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
		if (y < SUBPEL_TAPS) sm[0] += a * filter[y0_q4 & SUBPEL_MASK][y];
		if (y - 1 < SUBPEL_TAPS && y >= 1) sm[4] += a * filter[y0_q4 + y_step_q4 & SUBPEL_MASK][y - 1];
		if (y - 2 < SUBPEL_TAPS && y >= 2) sm[8] += a * filter[y0_q4 + 2 * y_step_q4 & SUBPEL_MASK][y - 2];
		if (y - 3 < SUBPEL_TAPS && y >= 3) sm[12] += a * filter[y0_q4 + 3 * y_step_q4 & SUBPEL_MASK][y - 3];
		
		++cashe;
		x_filter = filter[x0_q4 + x_step_q4 & SUBPEL_MASK];
		sum = 0;
		for (k = 0; k < SUBPEL_TAPS; ++k) sum += cashe[k] * x_filter[k];
		a = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
		if (y < SUBPEL_TAPS) sm[1] += a * filter[y0_q4 & SUBPEL_MASK][y];
		if (y - 1 < SUBPEL_TAPS && y >= 1) sm[5] += a * filter[y0_q4 + y_step_q4 & SUBPEL_MASK][y - 1];
		if (y - 2 < SUBPEL_TAPS && y >= 2) sm[9] += a * filter[y0_q4 + 2 * y_step_q4 & SUBPEL_MASK][y - 2];
		if (y - 3 < SUBPEL_TAPS && y >= 3) sm[13] += a * filter[y0_q4 + 3 * y_step_q4 & SUBPEL_MASK][y - 3];
		
		++cashe;
		x_filter = filter[x0_q4 + 2 * x_step_q4 & SUBPEL_MASK];
		sum = 0;
		for (k = 0; k < SUBPEL_TAPS; ++k) sum += cashe[k] * x_filter[k];
		a = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
		if (y < SUBPEL_TAPS) sm[2] += a * filter[y0_q4 & SUBPEL_MASK][y];
		if (y - 1 < SUBPEL_TAPS && y >= 1) sm[6] += a * filter[y0_q4 + y_step_q4 & SUBPEL_MASK][y - 1];
		if (y - 2 < SUBPEL_TAPS && y >= 2) sm[10] += a * filter[y0_q4 + 2 * y_step_q4 & SUBPEL_MASK][y - 2];
		if (y - 3 < SUBPEL_TAPS && y >= 3) sm[14] += a * filter[y0_q4 + 3 * y_step_q4 & SUBPEL_MASK][y - 3];
		
		++cashe;
		x_filter = filter[x0_q4 + 3 * x_step_q4 & SUBPEL_MASK];
		sum = 0;
		for (k = 0; k < SUBPEL_TAPS; ++k) sum += cashe[k] * x_filter[k];
		a = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
		if (y < SUBPEL_TAPS) sm[3] += a * filter[y0_q4 & SUBPEL_MASK][y];
		if (y - 1 < SUBPEL_TAPS && y >= 1) sm[7] += a * filter[y0_q4 + y_step_q4 & SUBPEL_MASK][y - 1];
		if (y - 2 < SUBPEL_TAPS && y >= 2) sm[11] += a * filter[y0_q4 + 2 * y_step_q4 & SUBPEL_MASK][y - 2];
		if (y - 3 < SUBPEL_TAPS && y >= 3) sm[15] += a * filter[y0_q4 + 3 * y_step_q4 & SUBPEL_MASK][y - 3];
		
		++y0;
		if (y0 > 0 && y0 < frame_h) ref_row += src_stride;
	}

	uint16_t* p = (uint16_t*)sm;
	p[0] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[0], FILTER_BITS), bd);
	p[1] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[1], FILTER_BITS), bd);
	p[2] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[2], FILTER_BITS), bd);
	p[3] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[3], FILTER_BITS), bd);
	p[4] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[4], FILTER_BITS), bd);
	p[5] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[5], FILTER_BITS), bd);
	p[6] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[6], FILTER_BITS), bd);
	p[7] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[7], FILTER_BITS), bd);
	p[8] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[8], FILTER_BITS), bd);
	p[9] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[9], FILTER_BITS), bd);
	p[10] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[10], FILTER_BITS), bd);
	p[11] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[11], FILTER_BITS), bd);
	p[12] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[12], FILTER_BITS), bd);
	p[13] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[13], FILTER_BITS), bd);
	p[14] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[14], FILTER_BITS), bd);
	p[15] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[15], FILTER_BITS), bd);
	
	*(long long*)(&dst[0 * dst_stride]) = *(long long*)p;
	*(long long*)(&dst[1 * dst_stride]) = *(long long*)(&p[4]);
	*(long long*)(&dst[2 * dst_stride]) = *(long long*)(&p[8]);
	*(long long*)(&dst[3 * dst_stride]) = *(long long*)(&p[12]);
}

__device__ static void cuda_highbd_convolve(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
		const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int bd, int y0, int frame_h) {
	int sm[16];
	memset(sm, 0, 16 * sizeof(int));
	
	assert(y_step_q4 <= 32);
	assert(x_step_q4 <= 32);
	
	int x, y;
	const uint16_t* ref_row = src - y0 * src_stride;
	y0 -= (SUBPEL_TAPS / 2 - 1);
	if (y0 >= frame_h) ref_row += (frame_h - 1) * src_stride;
	else if (y0 > 0) ref_row += y0 * src_stride;
	ref_row -= (SUBPEL_TAPS / 2 - 1);
	
	for (y = 0; y < 11; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < 4; ++x)
		{
			const uint16_t* const src_x = &ref_row[x];
			const int16_t* const x_filter = filter[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			int a = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
	
			int y_q4 = y0_q4;
			for (int j = 0; j < 4; ++j)
			{
				if (y - j < SUBPEL_TAPS && y >= j)
				{
					sm[j * 4 + x] += a * filter[y_q4 & SUBPEL_MASK][y - j];
				}
				y_q4 += y_step_q4;
			}
			x_q4 += x_step_q4;
		}
		++y0;
		if (y0 > 0 && y0 < frame_h) ref_row += src_stride;
	}
	
	for (x = 0; x < 4; ++x)
	{
		for (y = 0; y < 4; ++y)
		{
			dst[y * dst_stride] = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[y * 4 + x], FILTER_BITS), bd);
		}
		++dst;
	}
}

__device__ void cuda_vpx_highbd_convolve8_horiz_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                  const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                                  int w, int h, int bd) {
	(void)y0_q4;
	(void)y_step_q4;

	cuda_highbd_convolve_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_avg_horiz_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                      const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                                      int w, int h, int bd) {
	(void)y0_q4;
	(void)y_step_q4;

	cuda_highbd_convolve_avg_horiz(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_vert_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                 const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                                 int h, int bd) {
	(void)x0_q4;
	(void)x_step_q4;

	cuda_highbd_convolve_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_avg_vert_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                     const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                                                     int w, int h, int bd) {
	(void)x0_q4;
	(void)x_step_q4;

	cuda_highbd_convolve_avg_vert(src, src_stride, dst, dst_stride, filter, y0_q4, y_step_q4, w, h, bd);
}

__device__ void cuda_vpx_highbd_convolve8_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride, const InterpKernel* filter, 
											int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int bd, int y0, int frame_h)
{
	cuda_highbd_convolve(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, bd, y0, frame_h);
}

__device__ void cuda_vpx_highbd_convolve_avg_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
	const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd) {
	int x, y;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;
	(void)bd;

	for (y = 0; y < h; ++y)
	{
		for (x = 0; x < w; ++x) dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_highbd_convolve8_avg_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                                int h, int bd, int y0, int frame_h) {
	// Fixed size intermediate buffer places limits on parameters.
	assert(w <= 64);
	assert(h <= 64);

	//cuda_vpx_highbd_convolve8_c(src, src_stride, temp, 4, filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h, bd, y0, frame_h);

	int sm[16];
	memset(sm, 0, 16 * sizeof(int));
	const int intermediate_height = (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;
	
	assert(y_step_q4 <= 32);
	assert(x_step_q4 <= 32);

	int x, y;
	const uint16_t* ref_row = src - y0 * src_stride;
	y0 -= (SUBPEL_TAPS / 2 - 1);
	if (y0 >= frame_h) ref_row += (frame_h - 1) * src_stride;
	else if (y0 > 0) ref_row += y0 * src_stride;
	ref_row = ref_row - (SUBPEL_TAPS / 2 - 1);

	for (y = 0; y < intermediate_height; ++y)
	{
		int x_q4 = x0_q4;
		for (x = 0; x < w; ++x)
		{
			const uint16_t* const src_x = &ref_row[x];
			const int16_t* const x_filter = filter[x_q4 & SUBPEL_MASK];
			int k, sum = 0;
			for (k = 0; k < SUBPEL_TAPS; ++k) sum += src_x[k] * x_filter[k];
			int a = cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);

			int y_q4 = y0_q4;
			for (int j = 0; j < 4; ++j)
			{
				const int16_t* const y_filter = filter[y_q4 & SUBPEL_MASK];
				if (y - j < SUBPEL_TAPS && y >= j)
				{
					sm[j * 4 + x] += a * y_filter[y - j];
				}
				y_q4 += y_step_q4;
			}
			x_q4 += x_step_q4;
		}
		++y0;
		if (y0 > 0 && y0 < frame_h) ref_row += src_stride;
	}
	
	//cuda_vpx_highbd_convolve_avg_c(temp, 4, dst, dst_stride, NULL, 0, 0, 0, 0, w, h, bd);
	for (y = 0; y < h; ++y)
	{
		for (x = 0; x < w; ++x) 
			dst[x] = ROUND_POWER_OF_TWO(dst[x] + cuda_clip_pixel_highbd(ROUND_POWER_OF_TWO(sm[y * 4 + x], FILTER_BITS), bd), 1);
		dst += dst_stride;
	}
}

__device__ void cuda_vpx_highbd_convolve_copy_c(const uint16_t* src, ptrdiff_t src_stride, uint16_t* dst, ptrdiff_t dst_stride,
                                                const InterpKernel* filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w,
                                                int h, int bd) {
	int r;

	(void)filter;
	(void)x0_q4;
	(void)x_step_q4;
	(void)y0_q4;
	(void)y_step_q4;
	(void)bd;

	for (r = h; r > 0; --r)
	{
		memcpy(dst, src, w * sizeof(uint16_t));
		src += src_stride;
		dst += dst_stride;
	}
}

__device__ static void cuda_dec_build_inter_predictors_both(int subsampling, int y0,
	int x0, int bit_depth, const InterpKernel* kernel, int buf_stride, uint8_t* dst_buf, int16_t row, 
	int16_t col, uint8_t* ref_frame, int frame_height) {
	MV32 scaled_mv;
	int subpel_x, subpel_y;
	
	scaled_mv.row = row * (1 << (1 - subsampling));
	scaled_mv.col = col * (1 << (1 - subsampling));

	subpel_x = scaled_mv.col & SUBPEL_MASK;
	subpel_y = scaled_mv.row & SUBPEL_MASK;

	// Calculate the top left corner of the best matching block in the reference frame.
	x0 += scaled_mv.col >> SUBPEL_BITS;
	y0 += scaled_mv.row >> SUBPEL_BITS;
	

	y0 -= (SUBPEL_TAPS / 2 - 1);
	x0 -= (SUBPEL_TAPS / 2 - 1);
	
#if CONFIG_VP9_HIGHBITDEPTH
	shared_convolve_row(CONVERT_TO_SHORTPTR(ref_frame), buf_stride, CONVERT_TO_SHORTPTR(dst_buf), buf_stride, kernel,
		subpel_x, 16, subpel_y, 16, bit_depth, y0, frame_height, x0);
#else
	cuda_highbd_inter_predictor(buf_ptr, buf_stride, dst, buf_stride, subpel_x,
		subpel_y, sf, w, h, ref, kernel, xs, ys, bit_depth);
#endif
}

__device__ static void block_sum(int bd, int stride, uint16_t* dst_buf, tran_high_t* buf)
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			dst_buf[j * stride + i] = cuda_highbd_clip_pixel_add(
				dst_buf[j * stride + i], buf[j * stride + i], bd);
		}
	}
}

__device__ static void cuda_dec_build_inter_predictors_4x4_both(FrameInformation* fi, int16_t subsampling, int8_t interp_filter,
	uint8_t* alloc, uint8_t* ref_alloc, int16_t my, int16_t mx, int16_t y0, int16_t x0, int8_t plane, tran_high_t* residuals_alloc, int8_t skip) {

	uint8_t* y_buf = (uint8_t*)yv12_align_addr(alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
	uint8_t* u_buf = (uint8_t*)yv12_align_addr(alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
	uint8_t* v_buf = (uint8_t*)yv12_align_addr(alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
	
	uint8_t* buffer = (plane == 0 ? y_buf : (plane == 1 ? u_buf : v_buf));
	int stride = (plane == 0 ? fi->y_stride : fi->uv_stride);

	const InterpKernel* kernel = cuda_vp9_filter_kernels[interp_filter];
	
	uint8_t* ref_y_buf = (uint8_t*)yv12_align_addr(ref_alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
	uint8_t* ref_u_buf = (uint8_t*)yv12_align_addr(ref_alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
	uint8_t* ref_v_buf = (uint8_t*)yv12_align_addr(ref_alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);

	uint8_t* ref_buffer = (plane == 0 ? ref_y_buf : (plane == 1 ? ref_u_buf : ref_v_buf));

	uint8_t* dst_buf = buffer + y0 * stride + x0;
	
	int frame_height = (plane == 0 ? fi->y_crop_height : fi->uv_crop_height);

	cuda_dec_build_inter_predictors_both(subsampling, y0, x0, fi->bit_depth,  kernel, stride, dst_buf, my, mx,
		ref_buffer, frame_height);

	if (!skip)
	{
		tran_high_t* y_residuals = (tran_high_t*)yv12_align_addr(residuals_alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
		tran_high_t* u_residuals = (tran_high_t*)yv12_align_addr(residuals_alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
		tran_high_t* v_residuals = (tran_high_t*)yv12_align_addr(residuals_alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
	
		tran_high_t* residuals_buffer = plane == 0 ? y_residuals : (plane == 1 ? u_residuals : v_residuals);
		residuals_buffer = residuals_buffer + y0 * stride + x0;
	
		block_sum(fi->bit_depth, stride, CONVERT_TO_SHORTPTR(dst_buf), residuals_buffer);
	}
}

__global__ static void cuda_inter_4x4_both(uint8_t* alloc, uint8_t* frame_ref0, uint8_t* frame_ref1, uint8_t* frame_ref2, FrameInformation* fi,
	const int super_size, int8_t* block_settings, tran_high_t* residuals)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	uint8_t* converted_alloc = alloc;

#if CONFIG_VP9_HIGHBITDEPTH
	converted_alloc = CONVERT_TO_BYTEPTR(converted_alloc);
	
	if (i < super_size) {
		int k = block_settings[12 * i + 10] - LAST_FRAME;
		uint8_t* ref_alloc = (k == 0 ? CONVERT_TO_BYTEPTR(frame_ref0) : (k == 1 ? CONVERT_TO_BYTEPTR(frame_ref1) : CONVERT_TO_BYTEPTR(frame_ref2)));

		uint16_t* sq16 = (uint16_t*)&block_settings[12 * i + 2];
		uint16_t fq = ((uint16_t*)&block_settings[12 * i])[0];
		int8_t* fq8 = (int8_t*)&fq;
		
		const uint8_t plane = fq8[0];
		const uint16_t subsampling = (plane == 0 ? 0 : fi->chroma_subsampling);

		cuda_dec_build_inter_predictors_4x4_both(fi, subsampling, fq8[1], converted_alloc, ref_alloc,
			sq16[3], sq16[2], sq16[1], sq16[0], plane, residuals, block_settings[12 * i + 11]);
	}
#else

#endif
}

__host__ int createBuffers(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols, 
	int8_t* host_block_settings, FrameInformation* host_fi)
{
	YV12_BUFFER_CONFIG* src = &cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf;
	
	const int uv_border_h = src->border >> src->subsampling_y;
	const int uv_border_w = src->border >> src->subsampling_x;

	const int byte_alignment = cm->byte_alignment;
	const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
	const uint64_t yplane_size = (src->y_height + 2 * src->border) * (uint64_t)src->y_stride + byte_alignment;
	const uint64_t uvplane_size = (src->uv_height + 2 * uv_border_h) * (uint64_t)src->uv_stride + byte_alignment;
	
	host_fi->border = src->border;
	host_fi->size = src->frame_size;
	host_fi->y_stride = src->y_stride;
	host_fi->uv_stride = src->uv_stride;
	host_fi->uv_border_h = uv_border_h;
	host_fi->uv_border_w = uv_border_w;
	host_fi->yplane_size = yplane_size;
	host_fi->uvplane_size = uvplane_size;
	host_fi->bit_depth = cm->bit_depth;
	host_fi->vp9_byte_align = vp9_byte_align;
	host_fi->y_crop_width = src->y_crop_width;
	host_fi->y_crop_height = src->y_crop_height;
	host_fi->uv_crop_width = src->uv_crop_width;
	host_fi->uv_crop_height = src->uv_crop_height;
	host_fi->chroma_subsampling = pbi->tile_worker_data->xd.plane[1].subsampling_x;

	ModeInfoBuf strt_mi_buf;
	strt_mi_buf.bhl = MiBuf->bhl;
	strt_mi_buf.bwl = MiBuf->bwl;
	strt_mi_buf.mi_col = MiBuf->mi_col;
	strt_mi_buf.mi_row = MiBuf->mi_row;
	strt_mi_buf.mi = MiBuf->mi;
	int* strt_size_for_mb = size_for_mb;
	
	TileWorkerData* tile_data;
	int super_size = 0;
	int tile_row, mi_col, tile_col, mi_row;
	
	for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
		TileInfo tile;
		vp9_tile_set_row(&tile, cm, tile_row);
		for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
			for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
				const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
				tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
				vp9_tile_set_col(&tile, cm, col);
				MACROBLOCKD* const xd = &tile_data->xd;
				for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
					for (int i = 0; i < *size_for_mb; ++i) {
						if (is_inter_block(MiBuf->mi[0])) {
							const int bh = 1 << (*MiBuf->bhl - 1);
							const int bw = 1 << (*MiBuf->bwl - 1);
							int plane;
							if (MiBuf->mi[0]->sb_type < BLOCK_8X8) {
								for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
									int t_subsampling_x = xd->plane[plane].subsampling_x;
									int t_subsampling_y = xd->plane[plane].subsampling_y;

									const int num_4x4_w = (bw << 1) >> t_subsampling_x;
									const int num_4x4_h = (bh << 1) >> t_subsampling_y;

									int k = 0;
									for (int y = 0; y < num_4x4_h; ++y) {
										for (int x = 0; x < num_4x4_w; ++x) {
											const MV mv0 = average_split_mvs(&xd->plane[plane], MiBuf->mi[0], 0, k);
											
											host_block_settings[12 * super_size + 10] = MiBuf->mi[0]->ref_frame[0];
											host_block_settings[12 * super_size + 11] = MiBuf->mi[0]->skip;
											((int16_t*)&host_block_settings[12 * super_size])[3] = mv0.col;
											((int16_t*)&host_block_settings[12 * super_size])[4] = mv0.row;
											host_block_settings[12 * super_size + 1] = MiBuf->mi[0]->interp_filter;
											((int16_t*)&host_block_settings[12 * super_size])[2] = ((*MiBuf->mi_row * MI_SIZE * 8) >> (3 + t_subsampling_x)) + 4 * y;
											((int16_t*)&host_block_settings[12 * super_size])[1] = ((*MiBuf->mi_col * MI_SIZE * 8) >> (3 + t_subsampling_x)) + 4 * x;
											host_block_settings[12 * super_size] = plane;

											++super_size;
										}
									}

								}
							}
							else {
								for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
									int t_subsampling_x = xd->plane[plane].subsampling_x;
									int t_subsampling_y = xd->plane[plane].subsampling_y;
	
									const int num_4x4_w = (bw << 1) >> t_subsampling_x;
									const int num_4x4_h = (bh << 1) >> t_subsampling_y;
	
									
									for (int y = 0; y < num_4x4_h; ++y) {
										for (int x = 0; x < num_4x4_w; ++x) {
											host_block_settings[12 * super_size + 10] = MiBuf->mi[0]->ref_frame[0];
											host_block_settings[12 * super_size + 11] = MiBuf->mi[0]->skip;
											((int16_t*)&host_block_settings[12 * super_size])[3] = MiBuf->mi[0]->mv[0].as_mv.col;
											((int16_t*)&host_block_settings[12 * super_size])[4] = MiBuf->mi[0]->mv[0].as_mv.row;
											host_block_settings[12 * super_size + 1] = MiBuf->mi[0]->interp_filter;
											((int16_t*)&host_block_settings[12 * super_size])[2] = ((*MiBuf->mi_row * MI_SIZE * 8) >> (3 + t_subsampling_x)) + 4 * y;
											((int16_t*)&host_block_settings[12 * super_size])[1] = ((*MiBuf->mi_col * MI_SIZE * 8) >> (3 + t_subsampling_x)) + 4 * x;
											host_block_settings[12 * super_size] = plane;

											++super_size;
										}
									}
								}
							}
						}
	
						++MiBuf->mi_col;
						++MiBuf->mi;
						++MiBuf->mi_row;
						++MiBuf->bhl;
						++MiBuf->bwl;
					}
					++size_for_mb;
				}
			}
		}
	}

	MiBuf->bhl = strt_mi_buf.bhl;
	MiBuf->bwl = strt_mi_buf.bwl;
	MiBuf->mi_row = strt_mi_buf.mi_row;
	MiBuf->mi_col = strt_mi_buf.mi_col;
	MiBuf->mi = strt_mi_buf.mi;
	size_for_mb = strt_size_for_mb;
	
	return super_size;
}

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int cuda_inter_prediction(int n, double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
                          VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols, tran_high_t* residuals)
{
	cudaStream_t stream1, stream2, stream3, stream4, stream5;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream5, cudaStreamNonBlocking);
	
	uint8_t* cd_alloc, * frame_ref1, * frame_ref2, * frame_ref3;
	tran_high_t* cd_residuals;
	FrameInformation* cd_fi, * host_fi;
	int8_t* block_settings, * host_block_settings;
	
	uint8_t* frame_refs[3] = { cm->frame_refs[0].buf->buffer_alloc, cm->frame_refs[1].buf->buffer_alloc, cm->frame_refs[2].buf->buffer_alloc };
	uint8_t* alloc = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.buffer_alloc;
	
	cudaHostAlloc((void**)&host_block_settings, 3 * n / 16 * (12 * sizeof(int8_t)), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_fi, sizeof(FrameInformation), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	int super_size = createBuffers(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, host_block_settings,  host_fi);
	
	cudaMalloc((void**)&cd_alloc, host_fi->size);
	cudaMalloc((void**)&cd_residuals, host_fi->size * sizeof(tran_high_t));
	cudaMalloc((void**)&frame_ref1, host_fi->size);
	cudaMalloc((void**)&frame_ref2, host_fi->size);
	cudaMalloc((void**)&frame_ref3, host_fi->size);
	
	clock_t copy_begin = clock();
	//copy
	{
		cudaHostGetDevicePointer((void**)&cd_fi, (void*)host_fi, 0);
		cudaHostGetDevicePointer((void**)&block_settings, (void*)host_block_settings, 0);
		cudaMemcpyAsync(cd_alloc, alloc, host_fi->size, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(frame_ref1, frame_refs[0], host_fi->size, cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(frame_ref2, frame_refs[1], host_fi->size, cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(frame_ref3, frame_refs[2], host_fi->size, cudaMemcpyHostToDevice, stream4);
		cudaMemcpyAsync(cd_residuals, residuals, host_fi->size * sizeof(tran_high_t), cudaMemcpyHostToDevice, stream5);
		cudaDeviceSynchronize();
	}
	clock_t copy_end = clock();
	*gpu_copy = (double)(copy_end - copy_begin) / CLOCKS_PER_SEC;
	
	float elapsed = 0;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int blocksPerGrid = MAX(1, (super_size + interThreadsPerBlock - 1) / interThreadsPerBlock);
	cuda_inter_4x4_both <<<blocksPerGrid, interThreadsPerBlock>>> (cd_alloc, frame_ref1, frame_ref2, frame_ref3, cd_fi,
		super_size, block_settings, cd_residuals);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*gpu_run = ((double)elapsed) / 1000;
	
	cudaMemcpy(alloc, cd_alloc, host_fi->size, cudaMemcpyDeviceToHost);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);
	cudaStreamDestroy(stream5);
	
	//free
	{
		cudaFreeHost(host_block_settings);
		cudaFree(cd_alloc);
		cudaFree(frame_ref1);
		cudaFree(frame_ref2);
		cudaFree(frame_ref3);
		cudaFree(cd_residuals);
		cudaFreeHost(host_fi);
	}
	
	return 0;
}




//_________________________________________________________AV1____________________________________________________________________

inline __host__ __device__ void operator+=(int4& a, int4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

inline  __host__ __device__ int4 operator*(int4 b, int a)
{
	return make_int4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __host__ __device__ void operator+=(int2& a, int2 b)
{
	a.x += b.x;
	a.y += b.y;
}

inline  __host__ __device__ int2 operator*(int2 b, int a)
{
	return make_int2(a * b.x, a * b.y);
}

enum
{
	Vp9OutputAdd = 64,
	Vp9OutputShift = 7,
};

struct InterData
{
	ulonglong4 planes[3];
	ulonglong4 refplanes[3 * 3];
	//int4 scale[8];
	//GlobalMotionWarp gm_warp[RefCount];
	//int gm_warp_flags;
	//int pixel_max;
};

struct InterLaunchParams
{
	unsigned long wi_count;
	int pass_offset;
	int width_log2;
	int height_log2;
};

struct Vp9Block
{
	uint16_t x;
	uint16_t y;
	uint32_t plane : 2;
	uint32_t ref : 3;
	uint32_t filter : 4;
	uint32_t reserved0 : 23;
	int16_t mv_x;
	int16_t mv_y;
	uint32_t reserved1;
};

__device__ int4 filter_line(const int* ref, unsigned long long offset, int4 fkernel0, int4 fkernel1)
{
	const int shift = 8 * (offset & 2);
	offset >>= 2;
	uint4 l0;
	uint2 l1;
	l0.x = ref[offset + 0];
	l0.y = ref[offset + 1];
	l0.z = ref[offset + 2];
	l0.w = ref[offset + 3];
	l1.x = ref[offset + 4];
	l1.y = ref[offset + 5];
	l0.x = (l0.x >> shift) | ((l0.y << (24 - shift)) << 8);
	l0.y = (l0.y >> shift) | ((l0.z << (24 - shift)) << 8);
	l0.z = (l0.z >> shift) | ((l0.w << (24 - shift)) << 8);
	l0.w = (l0.w >> shift) | ((l1.x << (24 - shift)) << 8);
	l1.x = (l1.x >> shift) | ((l1.y << (24 - shift)) << 8);
	l1.y = l1.y >> shift;
	int4 sum = make_int4(0, 0, 0, 0);
	sum.x += fkernel0.x * (int)((l0.x >> 0) & 0xffff);
	sum.y += fkernel0.x * (int)((l0.x >> 16) & 0xffff);
	sum.z += fkernel0.x * (int)((l0.y >> 0) & 0xffff);
	sum.w += fkernel0.x * (int)((l0.y >> 16) & 0xffff);

	sum.x += fkernel0.y * (int)((l0.x >> 16) & 0xffff);
	sum.y += fkernel0.y * (int)((l0.y >> 0) & 0xffff);
	sum.z += fkernel0.y * (int)((l0.y >> 16) & 0xffff);
	sum.w += fkernel0.y * (int)((l0.z >> 0) & 0xffff);

	sum.x += fkernel0.z * (int)((l0.y >> 0) & 0xffff);
	sum.y += fkernel0.z * (int)((l0.y >> 16) & 0xffff);
	sum.z += fkernel0.z * (int)((l0.z >> 0) & 0xffff);
	sum.w += fkernel0.z * (int)((l0.z >> 16) & 0xffff);

	sum.x += fkernel0.w * (int)((l0.y >> 16) & 0xffff);
	sum.y += fkernel0.w * (int)((l0.z >> 0) & 0xffff);
	sum.z += fkernel0.w * (int)((l0.z >> 16) & 0xffff);
	sum.w += fkernel0.w * (int)((l0.w >> 0) & 0xffff);


	sum.x += fkernel1.x * (int)((l0.z >> 0) & 0xffff);
	sum.y += fkernel1.x * (int)((l0.z >> 16) & 0xffff);
	sum.z += fkernel1.x * (int)((l0.w >> 0) & 0xffff);
	sum.w += fkernel1.x * (int)((l0.w >> 16) & 0xffff);

	sum.x += fkernel1.y * (int)((l0.z >> 16) & 0xffff);
	sum.y += fkernel1.y * (int)((l0.w >> 0) & 0xffff);
	sum.z += fkernel1.y * (int)((l0.w >> 16) & 0xffff);
	sum.w += fkernel1.y * (int)((l1.x >> 0) & 0xffff);

	sum.x += fkernel1.z * (int)((l0.w >> 0) & 0xffff);
	sum.y += fkernel1.z * (int)((l0.w >> 16) & 0xffff);
	sum.z += fkernel1.z * (int)((l1.x >> 0) & 0xffff);
	sum.w += fkernel1.z * (int)((l1.x >> 16) & 0xffff);

	sum.x += fkernel1.w * (int)((l0.w >> 16) & 0xffff);
	sum.y += fkernel1.w * (int)((l1.x >> 0) & 0xffff);
	sum.z += fkernel1.w * (int)((l1.x >> 16) & 0xffff);
	sum.w += fkernel1.w * (int)((l1.y >> 0) & 0xffff);
	sum.x = (sum.x + Vp9OutputAdd) >> Vp9OutputShift;
	sum.y = (sum.y + Vp9OutputAdd) >> Vp9OutputShift;
	sum.z = (sum.z + Vp9OutputAdd) >> Vp9OutputShift;
	sum.w = (sum.w + Vp9OutputAdd) >> Vp9OutputShift;
	return sum;
}

__global__ void inter_main_av1(const Vp9Block* pred_blocks, uint8_t* frame, uint8_t* d_frame, InterData* frame_params, InterLaunchParams* params)
{
	int* src_frame = (int*)frame;
	int* dst_frame = (int*)d_frame;
	//using Consts = BitDepthConsts<T, 0>;
	const int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= params->wi_count)
		return;

	const int w_log = params->width_log2;
	const int h_log = params->height_log2;
	const int subblock = thread & ((1 << (w_log + h_log)) - 1);
	const int block_index = params->pass_offset + (thread >> (w_log + h_log));
	Vp9Block block = pred_blocks[block_index];

	int x = /*Base::SubblockW*/(block.x + (subblock & ((1 << w_log) - 1)));
	int y = /*Base::SubblockH*/(block.y + (subblock >> w_log));

	const int plane = block.plane;
	const int refplane = block.ref * 3 + plane;
	const unsigned long long ref_offset = frame_params->refplanes[refplane].y;
	const int ref_stride = frame_params->refplanes[refplane].x;
	const int ref_w = frame_params->refplanes[refplane].z;
	const int ref_h = frame_params->refplanes[refplane].w;

	int mvx = x + (block.mv_x >> SUBPEL_BITS) - 3;
	int mvy = y + (block.mv_y >> SUBPEL_BITS) - 3;

	mvx = cuda_clamp(mvx, -11, ref_w) << /*Consts::Hbd*/1;

	const int filter_h = block.mv_x & SUBPEL_MASK;
	const int filter_v = block.mv_y & SUBPEL_MASK;

	int4 kernel_h0 = cuda_vp9_filter_kernels_int4[block.filter][filter_h][0];
	int4 kernel_h1 = cuda_vp9_filter_kernels_int4[block.filter][filter_h][1];

	int4 output[4] = { make_int4(0, 0, 0, 0),
						make_int4(0, 0, 0, 0),
						make_int4(0, 0, 0, 0),
						make_int4(0, 0, 0, 0) };

	int4 l;
	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 0, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;

	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 1, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;

	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 2, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;

	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 3, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 4, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 5, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 6, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 7, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 8, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 9, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	l = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 10, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	const int output_stride = frame_params->planes[plane].x;
	const unsigned long long output_offset = frame_params->planes[plane].y + (x << 1) + y * output_stride;
	for (int i = 0; i < 4; ++i)
	{
		output[i].x = cuda_clamp((output[i].x + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);
		output[i].y = cuda_clamp((output[i].y + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);
		output[i].z = cuda_clamp((output[i].z + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);
		output[i].w = cuda_clamp((output[i].w + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);

		const int ofs = (output_offset + i * output_stride) >> 2;
		dst_frame[ofs] = output[i].x | (output[i].y << 16);
		dst_frame[ofs + 1] = output[i].z | (output[i].w << 16);
	}
}

__device__ int filter_line_col(const int* ref, unsigned long long offset, int4 fkernel0, int4 fkernel1)
{
	const int shift = 8 * (offset & 2);
	offset >>= 2;
	uint4 l0;
	int l1;
	l0.x = ref[offset + 0];
	l0.y = ref[offset + 1];
	l0.z = ref[offset + 2];
	l0.w = ref[offset + 3];
	l1 = ref[offset + 4];

	l0.x = (l0.x >> shift) | ((l0.y << (24 - shift)) << 8);
	l0.y = (l0.y >> shift) | ((l0.z << (24 - shift)) << 8);
	l0.z = (l0.z >> shift) | ((l0.w << (24 - shift)) << 8);
	l0.w = (l0.w >> shift) | ((l1 << (24 - shift)) << 8);

	int sum = 0;
	sum += fkernel0.x * (int)((l0.x >> 0) & 0xffff);
	sum += fkernel0.y * (int)((l0.x >> 16) & 0xffff);
	sum += fkernel0.z * (int)((l0.y >> 0) & 0xffff);
	sum += fkernel0.w * (int)((l0.y >> 16) & 0xffff);


	sum += fkernel1.x * (int)((l0.z >> 0) & 0xffff);
	sum += fkernel1.y * (int)((l0.z >> 16) & 0xffff);
	sum += fkernel1.z * (int)((l0.w >> 0) & 0xffff);
	sum += fkernel1.w * (int)((l0.w >> 16) & 0xffff);

	sum = (sum + Vp9OutputAdd) >> Vp9OutputShift;
	return sum;
}

__global__ void inter_main_av1_col(const Vp9Block* pred_blocks, uint8_t* frame, uint8_t* d_frame, InterData* frame_params, InterLaunchParams* params)
{
	int* src_frame = (int*)frame;
	int* dst_frame = (int*)d_frame;
	//using Consts = BitDepthConsts<T, 0>;
	const int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= params->wi_count)
		return;

	const int w_log = params->width_log2;
	const int h_log = params->height_log2;
	const int subblock = thread & ((1 << (w_log + h_log)) - 1);
	const int block_index = params->pass_offset + (thread >> (w_log + h_log));
	Vp9Block block = pred_blocks[block_index / 4];

	int x = /*Base::SubblockW*/(block.x + (subblock & ((1 << w_log) - 1))) + threadIdx.x % 4;
	int write_x = /*Base::SubblockW*/(block.x + (subblock & ((1 << w_log) - 1)));
	int y = /*Base::SubblockH*/(block.y + (subblock >> w_log));

	const int plane = block.plane;
	const int refplane = block.ref * 3 + plane;
	const unsigned long long ref_offset = frame_params->refplanes[refplane].y;
	const int ref_stride = frame_params->refplanes[refplane].x;
	const int ref_w = frame_params->refplanes[refplane].z;
	const int ref_h = frame_params->refplanes[refplane].w;

	int mvx = x + (block.mv_x >> SUBPEL_BITS) - 3;
	int mvy = y + (block.mv_y >> SUBPEL_BITS) - 3;

	mvx = cuda_clamp(mvx, -11, ref_w) << /*Consts::Hbd*/1;

	const int filter_h = block.mv_x & SUBPEL_MASK;
	const int filter_v = block.mv_y & SUBPEL_MASK;

	int4 kernel_h0 = cuda_vp9_filter_kernels_int4[block.filter][filter_h][0];
	int4 kernel_h1 = cuda_vp9_filter_kernels_int4[block.filter][filter_h][1];

	int output[4] = { 0, 0, 0, 0 };

	int l;
	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 0, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;

	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 1, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;

	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 2, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;

	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 3, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 4, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 5, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 6, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 7, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output[0] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 8, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output[1] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 9, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output[2] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	l = filter_line_col(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 10, 0, ref_h - 1), kernel_h0, kernel_h1);
	output[3] += l * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	const int output_stride = frame_params->planes[plane].x;
	const unsigned long long output_offset = frame_params->planes[plane].y + (write_x << 1) + y * output_stride;
	__shared__ int cashe[interThreadsPerBlock];
	for (int i = 0; i < 4; ++i)
	{
		output[i] = cuda_clamp((output[i] + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);

		cashe[threadIdx.x] = output[i];

		const int ofs = (output_offset + i * output_stride) >> 2;
		__syncthreads();
		if (threadIdx.x % 4 == 0)
		{
			dst_frame[ofs] = cashe[threadIdx.x] | (cashe[threadIdx.x + 1] << 16);
			dst_frame[ofs + 1] = cashe[threadIdx.x + 2] | (cashe[threadIdx.x + 3] << 16);
		}
		__syncthreads();
	}
}

__global__ void inter_main_av1_row(const Vp9Block* pred_blocks, uint8_t* frame, uint8_t* d_frame, InterData* frame_params, InterLaunchParams* params)
{
	int* src_frame = (int*)frame;
	int* dst_frame = (int*)d_frame;
	//using Consts = BitDepthConsts<T, 0>;
	const int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread >= params->wi_count)
		return;

	const int w_log = params->width_log2;
	const int h_log = params->height_log2;
	const int subblock = thread & ((1 << (w_log + h_log)) - 1);
	const int block_index = params->pass_offset + (thread >> (w_log + h_log));
	Vp9Block block = pred_blocks[block_index / 4];

	int x = /*Base::SubblockW*/(block.x + (subblock & ((1 << w_log) - 1)));
	int y = /*Base::SubblockH*/(block.y + (subblock >> w_log)) + threadIdx.x % 4;

	const int plane = block.plane;
	const int refplane = block.ref * 3 + plane;
	const unsigned long long ref_offset = frame_params->refplanes[refplane].y;
	const int ref_stride = frame_params->refplanes[refplane].x;
	const int ref_w = frame_params->refplanes[refplane].z;
	const int ref_h = frame_params->refplanes[refplane].w;

	int mvx = x + (block.mv_x >> SUBPEL_BITS) - 3;
	int mvy = y + (block.mv_y >> SUBPEL_BITS) - 3;

	mvx = cuda_clamp(mvx, -11, ref_w) << /*Consts::Hbd*/1;

	const int filter_h = block.mv_x & SUBPEL_MASK;
	const int filter_v = block.mv_y & SUBPEL_MASK;

	int4 kernel_h0 = cuda_vp9_filter_kernels_int4[block.filter][filter_h][0];
	int4 kernel_h1 = cuda_vp9_filter_kernels_int4[block.filter][filter_h][1];

	int4 output = make_int4(0, 0, 0, 0);

	__shared__ int4 l[3 * interThreadsPerBlock];
	l[12 * (threadIdx.x / 4) + threadIdx.x % 4] = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 0, 0, ref_h - 1), kernel_h0, kernel_h1);
	l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 4] = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 4, 0, ref_h - 1), kernel_h0, kernel_h1);
	l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 8] = filter_line(src_frame, ref_offset + mvx + ref_stride * cuda_clamp(mvy + 8, 0, ref_h - 1), kernel_h0, kernel_h1);
	__syncthreads();

	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].x;
	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 1] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].y;
	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 2] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].z;
	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 3] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][0].w;

	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 4] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].x;
	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 5] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].y;
	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 6] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].z;
	output += l[12 * (threadIdx.x / 4) + threadIdx.x % 4 + 7] * cuda_vp9_filter_kernels_int4[block.filter][filter_v][1].w;

	const int output_stride = frame_params->planes[plane].x;
	const unsigned long long output_offset = frame_params->planes[plane].y + (x << 1) + y * output_stride;

	output.x = cuda_clamp((output.x + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);
	output.y = cuda_clamp((output.y + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);
	output.z = cuda_clamp((output.z + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);
	output.w = cuda_clamp((output.w + Vp9OutputAdd) >> Vp9OutputShift, 0, 1023);

	const int ofs = output_offset >> 2;
	dst_frame[ofs] = output.x | (output.y << 16);
	dst_frame[ofs + 1] = output.z | (output.w << 16);
}

__host__ int createBuffers_av1(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols,
	Vp9Block* host_block_settings, FrameInformation* host_fi)
{
	YV12_BUFFER_CONFIG* src = &cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf;

	const int uv_border_h = src->border >> src->subsampling_y;
	const int uv_border_w = src->border >> src->subsampling_x;

	const int byte_alignment = cm->byte_alignment;
	const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
	const uint64_t yplane_size = (src->y_height + 2 * src->border) * (uint64_t)src->y_stride + byte_alignment;
	const uint64_t uvplane_size = (src->uv_height + 2 * uv_border_h) * (uint64_t)src->uv_stride + byte_alignment;

	host_fi->border = src->border;
	host_fi->size = src->frame_size;
	host_fi->y_stride = src->y_stride;
	host_fi->uv_stride = src->uv_stride;
	host_fi->uv_border_h = uv_border_h;
	host_fi->uv_border_w = uv_border_w;
	host_fi->yplane_size = yplane_size;
	host_fi->uvplane_size = uvplane_size;
	host_fi->bit_depth = cm->bit_depth;
	host_fi->vp9_byte_align = vp9_byte_align;
	host_fi->y_crop_width = src->y_crop_width;
	host_fi->y_crop_height = src->y_crop_height;
	host_fi->uv_crop_width = src->uv_crop_width;
	host_fi->uv_crop_height = src->uv_crop_height;
	host_fi->chroma_subsampling = pbi->tile_worker_data->xd.plane[1].subsampling_x;

	ModeInfoBuf strt_mi_buf;
	strt_mi_buf.bhl = MiBuf->bhl;
	strt_mi_buf.bwl = MiBuf->bwl;
	strt_mi_buf.mi_col = MiBuf->mi_col;
	strt_mi_buf.mi_row = MiBuf->mi_row;
	strt_mi_buf.mi = MiBuf->mi;
	int* strt_size_for_mb = size_for_mb;

	TileWorkerData* tile_data;
	int super_size = 0;
	int tile_row, mi_col, tile_col, mi_row;
	int t_both = 0;

	for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
		TileInfo tile;
		vp9_tile_set_row(&tile, cm, tile_row);
		for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
			for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
				const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
				tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
				vp9_tile_set_col(&tile, cm, col);
				MACROBLOCKD* const xd = &tile_data->xd;
				for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
					for (int i = 0; i < *size_for_mb; ++i) {
						if (is_inter_block(MiBuf->mi[0])) {
							const int bh = 1 << (*MiBuf->bhl - 1);
							const int bw = 1 << (*MiBuf->bwl - 1);
							int plane;
							if (MiBuf->mi[0]->sb_type < BLOCK_8X8) {
								for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
									int t_subsampling_x = xd->plane[plane].subsampling_x;
									int t_subsampling_y = xd->plane[plane].subsampling_y;

									const int num_4x4_w = (bw << 1) >> t_subsampling_x;
									const int num_4x4_h = (bh << 1) >> t_subsampling_y;

									int k = 0;
									for (int y = 0; y < num_4x4_h; ++y) {
										for (int x = 0; x < num_4x4_w; ++x) {
											const MV mv0 = average_split_mvs(&xd->plane[plane], MiBuf->mi[0], 0, k);

											//host_ref_frame[1 * t_both + 0] = MiBuf->mi[0]->ref_frame[0];
											//host_block_settings[8 * t_both + 6] = MiBuf->mi[0]->mv[0].as_mv.col;
											//host_block_settings[8 * t_both + 7] = MiBuf->mi[0]->mv[0].as_mv.row;
											//host_block_settings[8 * t_both + 3] = MiBuf->mi[0]->interp_filter;
											//host_block_settings[8 * t_both + 5] = *MiBuf->mi_row;
											//host_block_settings[8 * t_both + 4] = *MiBuf->mi_col;
											//host_block_settings[8 * t_both + 0] = x;
											//host_block_settings[8 * t_both + 1] = y;
											//host_block_settings[8 * t_both + 2] = plane;

											++t_both;
										}
									}

								}
							}
							else {
								for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
									int t_subsampling_x = xd->plane[plane].subsampling_x;
									int t_subsampling_y = xd->plane[plane].subsampling_y;

									const int num_4x4_w = (bw << 1) >> t_subsampling_x;
									const int num_4x4_h = (bh << 1) >> t_subsampling_y;

									for (int y = 0; y < num_4x4_h; ++y) {
										for (int x = 0; x < num_4x4_w; ++x) {
											host_block_settings[t_both].ref = MiBuf->mi[0]->ref_frame[0];
											host_block_settings[t_both].mv_x = MiBuf->mi[0]->mv[0].as_mv.col * (1 << (1 - t_subsampling_x));
											host_block_settings[t_both].mv_y = MiBuf->mi[0]->mv[0].as_mv.row * (1 << (1 - t_subsampling_y));
											host_block_settings[t_both].filter = MiBuf->mi[0]->interp_filter;
											host_block_settings[t_both].y = ((*MiBuf->mi_row * MI_SIZE * 8) >> (3 + t_subsampling_x)) + 4 * y;
											host_block_settings[t_both].x = ((*MiBuf->mi_col * MI_SIZE * 8) >> (3 + t_subsampling_x)) + 4 * x;
											host_block_settings[t_both].plane = plane;

											++super_size;
											++t_both;
										}
									}
								}
							}
						}

						++MiBuf->mi_col;
						++MiBuf->mi;
						++MiBuf->mi_row;
						++MiBuf->bhl;
						++MiBuf->bwl;
					}
					++size_for_mb;
				}
			}
		}
	}

	MiBuf->bhl = strt_mi_buf.bhl;
	MiBuf->bwl = strt_mi_buf.bwl;
	MiBuf->mi_row = strt_mi_buf.mi_row;
	MiBuf->mi_col = strt_mi_buf.mi_col;
	MiBuf->mi = strt_mi_buf.mi;
	size_for_mb = strt_size_for_mb;

	return super_size;
}

int cuda_inter_av1(int n, double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
	VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols)
{
	Vp9Block* pred_blocks, * host_pred_blocks;
	uint8_t* dst_frame, * src_frame, * host_src_frame;
	InterData* frame_params, * host_frame_params;
	InterLaunchParams* params, * host_params;
	FrameInformation host_fi;

	uint8_t* frame_refs[3] = { cm->frame_refs[0].buf->buffer_alloc, cm->frame_refs[1].buf->buffer_alloc, cm->frame_refs[2].buf->buffer_alloc };
	uint8_t* alloc = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.buffer_alloc;

	size_t a = sizeof(Vp9Block);

	cudaHostAlloc((void**)&host_pred_blocks, 3 * n / 16 * sizeof(Vp9Block), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_frame_params, sizeof(InterData), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_params, sizeof(InterLaunchParams), cudaHostAllocWriteCombined | cudaHostAllocMapped);

	int super_size = 4 * createBuffers_av1(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, host_pred_blocks, &host_fi);

	host_params->wi_count = super_size;
	host_params->height_log2 = 0;
	host_params->pass_offset = 0;
	host_params->width_log2 = 0;


	unsigned long long y_buf_offset = 2 * ((host_fi.border * host_fi.y_stride) + host_fi.border);
	unsigned long long u_buf_offset = 2 * (host_fi.yplane_size + (host_fi.uv_border_h * host_fi.uv_stride) + host_fi.uv_border_w);
	unsigned long long v_buf_offset = 2 * (host_fi.yplane_size + host_fi.uvplane_size + (host_fi.uv_border_h * host_fi.uv_stride) + host_fi.uv_border_w);

	host_frame_params->refplanes[0] = { (unsigned long long)(2 * host_fi.y_stride), y_buf_offset, host_fi.y_crop_width, host_fi.y_crop_height };
	host_frame_params->refplanes[1] = { (unsigned long long)(2 * host_fi.uv_stride),u_buf_offset, host_fi.uv_crop_width, host_fi.uv_crop_height };
	host_frame_params->refplanes[2] = { (unsigned long long)(2 * host_fi.uv_stride),v_buf_offset,host_fi.uv_crop_width, host_fi.uv_crop_height };

	y_buf_offset += 2 * host_fi.size;
	u_buf_offset += 2 * host_fi.size;
	v_buf_offset += 2 * host_fi.size;

	host_frame_params->refplanes[3] = { (unsigned long long)(2 * host_fi.y_stride),  y_buf_offset, host_fi.y_crop_width, host_fi.y_crop_height };
	host_frame_params->refplanes[4] = { (unsigned long long)(2 * host_fi.uv_stride), u_buf_offset, host_fi.uv_crop_width, host_fi.uv_crop_height };
	host_frame_params->refplanes[5] = { (unsigned long long)(2 * host_fi.uv_stride),  v_buf_offset,host_fi.uv_crop_width, host_fi.uv_crop_height };

	y_buf_offset += 2 * host_fi.size;
	u_buf_offset += 2 * host_fi.size;
	v_buf_offset += 2 * host_fi.size;

	host_frame_params->refplanes[6] = { (unsigned long long)(2 * host_fi.y_stride),  y_buf_offset,host_fi.y_crop_width, host_fi.y_crop_height };
	host_frame_params->refplanes[7] = { (unsigned long long)(2 * host_fi.uv_stride), u_buf_offset, host_fi.uv_crop_width, host_fi.uv_crop_height };
	host_frame_params->refplanes[8] = { (unsigned long long)(2 * host_fi.uv_stride),  v_buf_offset,host_fi.uv_crop_width, host_fi.uv_crop_height };

	y_buf_offset = 2 * ((host_fi.border * host_fi.y_stride) + host_fi.border);
	u_buf_offset = 2 * (host_fi.yplane_size + (host_fi.uv_border_h * host_fi.uv_stride) + host_fi.uv_border_w);
	v_buf_offset = 2 * (host_fi.yplane_size + host_fi.uvplane_size + (host_fi.uv_border_h * host_fi.uv_stride) + host_fi.uv_border_w);

	host_frame_params->planes[0] = { (unsigned long long)(2 * host_fi.y_stride),  y_buf_offset,0, 0 };
	host_frame_params->planes[1] = { (unsigned long long)(2 * host_fi.uv_stride), u_buf_offset, 0, 0 };
	host_frame_params->planes[2] = { (unsigned long long)(2 * host_fi.uv_stride), v_buf_offset,0, 0 };

	cudaMalloc((void**)&src_frame, 3 * host_fi.size);
	cudaMalloc((void**)&dst_frame, host_fi.size);
	host_src_frame = (uint8_t*)malloc(3 * host_fi.size);

	//copy
	{
		cudaHostGetDevicePointer((void**)&frame_params, (void*)host_frame_params, 0);
		cudaHostGetDevicePointer((void**)&params, (void*)host_params, 0);
		cudaHostGetDevicePointer((void**)&pred_blocks, (void*)host_pred_blocks, 0);
		cudaMemcpy(host_src_frame, frame_refs[0], host_fi.size, cudaMemcpyHostToHost);
		cudaMemcpy(dst_frame, alloc, host_fi.size, cudaMemcpyHostToHost);
		cudaMemcpy(host_src_frame + host_fi.size, frame_refs[1], host_fi.size, cudaMemcpyHostToHost);
		cudaMemcpy(host_src_frame + 2 * host_fi.size, frame_refs[2], host_fi.size, cudaMemcpyHostToHost);
		cudaMemcpy(src_frame, host_src_frame, 3 * host_fi.size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}

	int blocksPerGrid = MAX(1, (super_size + interThreadsPerBlock - 1) / interThreadsPerBlock);
	inter_main_av1_row <<<blocksPerGrid, interThreadsPerBlock >>> (pred_blocks, src_frame, dst_frame, frame_params, params);

	cudaDeviceSynchronize();

	cudaMemcpy(alloc, dst_frame, host_fi.size, cudaMemcpyHostToHost);

	//free
	{
		cudaFree(src_frame);
		cudaFree(dst_frame);
		cudaFreeHost(host_pred_blocks);
		cudaFreeHost(host_frame_params);
		cudaFreeHost(host_params);
		free(host_src_frame);
	}

	return 0;
}