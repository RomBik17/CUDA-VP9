#pragma once
#include "vp9/common/vp9_blockd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "time.h"

#define yv12_align_addr(addr, align) \
  (void *)(((size_t)(addr) + ((align)-1)) & (size_t) - (align))

typedef struct frame_buffer {
	tran_high_t* residuals;
	tran_high_t* plane_residuals[MAX_MB_PLANE];
	tran_low_t** dqcoeff;
	int* eob;
	int* plane_eob[MAX_MB_PLANE];
} frameBuf;

typedef struct intra_buf {
	int* bhl;
	int* bwl;
	int* block_settings;
	int t_subsampling_x;
	int t_subsampling_y;
	int* mb_to_left_edge;
	int* mb_to_right_edge;
	int* mb_to_top_edge;
	int* mb_to_bottom_edge;
	int bit_depth;
	BLOCK_SIZE* sb_type;
	MV_REFERENCE_FRAME* ref_frame;
	int16_t* mv;
} IntraBuf;

typedef struct frame_info {
	uint16_t	border;
	size_t		size;
	uint16_t	y_stride;
	uint16_t	uv_stride;
	uint16_t	uv_border_h;
	uint16_t	uv_border_w;
	uint64_t	yplane_size;
	uint64_t	uvplane_size;
	uint16_t	bit_depth;
	int			vp9_byte_align;
	uint16_t	y_crop_width;
	uint16_t	y_crop_height;
	uint16_t	uv_crop_width;
	uint16_t	uv_crop_height;
	uint16_t	chroma_subsampling;
} FrameInformation;

typedef struct log_buf {
	MODE_INFO** mi;
	int* mi_row;
	int* mi_col;
	int* bwl;
	int* bhl;
} ModeInfoBuf;