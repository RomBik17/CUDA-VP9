
#include "intra_cuda_kernel.cuh"

__device__ void cuda_vpx_memset16(uint16_t* dest, int val, size_t length) {
	size_t i;
	uint16_t* dest16 = dest;
	for (i = 0; i < length; i++) *dest16++ = val;
}

__device__ int intra_clamp(int value, int low, int high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ uint16_t intra_clip_pixel_highbd(int val, int bd)
{
	switch (bd)
	{
	case 8:
	default: return (uint16_t)intra_clamp(val, 0, 255);
	case 10: return (uint16_t)intra_clamp(val, 0, 1023);
	case 12: return (uint16_t)intra_clamp(val, 0, 4095);
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

__device__ static uint16_t intra_highbd_clip_pixel_add(int dest, tran_high_t trans, int bd) {
	trans = cuda_highbd_check_range(trans, bd);
	return intra_clip_pixel_highbd(dest + (int)trans, bd);
}

__device__ static void highbd_d207_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* left,
		int ty, int tx) {
	int r, c;

	dst += ty * stride;

	
	for (r = ty; r < ty + 4; ++r) {
		for (c = tx; c < tx + 4; ++c)
		{
			int row = r, col = c;
			int h = bs - 1 - row;
			if (h * 2 < col)
			{
				col -= h * 2;
				row = bs - 1;
			}
			else
			{
				row += col / 2;
				col %= 2;
			}
			
			if (col == 0)
			{
				if (row == bs - 1) dst[c] = left[bs - 1];
				else dst[c] = AVG2(left[row], left[row + 1]);
			}
			else if (col == 1)
			{
				if (row == bs - 1) dst[c] = left[bs - 1];
				else if (row == bs - 2) dst[c] = AVG3(left[bs - 2], left[bs - 1], left[bs - 1]);
				else dst[c] = AVG3(left[row], left[row + 1], left[row + 2]);
			}
			else  dst[c] = left[bs - 1];
		}
		dst += stride;
	}
}

__device__ static void highbd_d63_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* above, int ty, int tx) {
	int r, c, i;
	int size;
	
	for (r = ty, size = bs - ty / 2; r < ty + 4; r += 2, --size) {
		for(c = tx; c < size && c < tx + 4; ++c)
		{
			i = c + (r >> 1);
			dst[r * stride + c] = AVG2(above[i], above[i + 1]);
			dst[(r + 1) * stride + c] = AVG3(above[i], above[i + 1], above[i + 2]);
		}
		
		for (i = c; i < bs && i < tx + 4; i++)
		{
			dst[r * stride + i] = above[bs - 1];
			dst[(r + 1) * stride + i] = above[bs - 1];
		}
	}
}

__device__ static void highbd_d45_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* above, int ty, int tx) {
	int c, i, r, size;

	dst += ty * stride;
	
	for (r = ty, size = bs - 1 - ty; r < ty + 4; ++r, --size) {
		for(c = tx; (c < size) && c < tx + 4; ++c)
		{
			if (r + c == bs - 1) dst[c] = above[bs - 1];
			else dst[c] = AVG3(above[r + c], above[r + c + 1], above[r + c + 2]);
		}
		for (i = c; i < size + r + 1 && i < tx + 4; ++i) dst[i] = above[bs - 1];
		dst += stride;
	}
}

__device__ static void highbd_d117_predictor(uint16_t* dst, ptrdiff_t stride, const uint16_t* above, const uint16_t* left,
		int ty, int tx) {
	int r, c;

	dst += ty * stride;

	// the rest of the block
	for (r = ty; r < ty + 4; ++r) {
		for (c = tx; c < tx + 4; ++c)
		{
			int row = r, col = c;
			if (col * 2 < row)
			{
				row -= col * 2;
				col = 0;
			}
			else
			{
				col -= row / 2;
				row %= 2;
			}
			
			if (row == 0) dst[c] = AVG2(above[col - 1], above[col]);
			else if (row == 1)
			{
				if (col == 0) dst[c] = AVG3(left[0], above[-1], above[0]);
				else dst[c] = AVG3(above[col - 2], above[col - 1], above[col]);
			}
			else if (row == 2) dst[c] = AVG3(above[-1], left[0], left[1]);
			else dst[c] = AVG3(left[row - 3], left[row - 2], left[row - 1]);
		}
		dst += stride;
	}
}

__device__ static void highbd_d135_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* above, const uint16_t* left, 
		int ty, int tx) {
	int i;
	
	for (i = 0; i < 4; ++i) {
		for(int j = 0; j < 4; ++j)
		{
			int k = bs - 1 - (ty + i) + tx + j;
			if (k < bs - 2) dst[i * stride + j] = AVG3(left[bs - 3 - k], left[bs - 2 - k], left[bs - 1 - k]);
			else if (k == bs - 2) dst[i * stride + j] = AVG3(above[-1], left[0], left[1]);
			else if (k == bs - 1) dst[i * stride + j] = AVG3(left[0], above[-1], above[0]);
			else if (k == bs - 0) dst[i * stride + j] = AVG3(above[-1], above[0], above[1]);
			else dst[i * stride + j] = AVG3(above[k - bs - 1], above[k - bs], above[k - bs + 1]);
		}
	}
}

__device__ static void highbd_d153_predictor(uint16_t* dst, ptrdiff_t stride, const uint16_t* above, const uint16_t* left, 
		int ty, int tx) {
	int r, c;

	dst += ty * stride;
	
	for (r = ty; r < ty + 4; ++r) {
		for (c = tx; c < tx + 4; ++c)
		{
			int row = r, col = c;
			if (row * 2 < col)
			{
				col -= row * 2;
				row = 0;
			}
			else
			{
				row -= col / 2;
				col %= 2;
			}
		
			if (col == 0)
			{
				if (row == 0) dst[c] = AVG2(above[-1], left[0]);
				else dst[c] = AVG2(left[row - 1], left[row]);
			}
			else if (col == 1)
			{
			    if (row == 0) dst[c] = AVG3(left[0], above[-1], above[0]);
			    else if (row == 1) dst[c] = AVG3(above[-1], left[0], left[1]);
				else dst[c] = AVG3(left[row - 2], left[row - 1], left[row]);
			}
			else dst[c] = AVG3(above[col - 3], above[col - 2], above[col - 1]);
		}
		dst += stride;
	}
}

__device__ static void highbd_v_predictor(uint16_t* dst, ptrdiff_t stride, const uint16_t* above) {
	for (int r = 0; r < 4; ++r) {
		memcpy(dst, above, 4 * sizeof(uint16_t));
		dst += stride;
	}
}

__device__ static void highbd_h_predictor(uint16_t* dst, ptrdiff_t stride, const uint16_t* left) {
	for (int r = 0; r < 4; ++r) {
		cuda_vpx_memset16(dst, left[r], 4);
		dst += stride;
	}
}

__device__ static void highbd_tm_predictor(uint16_t* dst, ptrdiff_t stride, const uint16_t* above, 
		const uint16_t* left, int bd, int ty, int tx) {
	int r, c;
	int ytop_left = above[-1];
	(void)bd;

	above += tx;
	left += ty;
	
	for (r = 0; r < 4; ++r) {
		for (c = 0; c < 4; ++c)
			dst[c] = intra_clip_pixel_highbd(left[r] + above[c] - ytop_left, bd);
		dst += stride;
	}
}

__device__ static void highbd_dc_128_predictor(uint16_t* dst, ptrdiff_t stride, int bs, int bd) {
	for (int r = 0; r < 4; ++r) {
		cuda_vpx_memset16(dst, 128 << (bd - 8), 4);
		dst += stride;
	}
}

__device__ static void highbd_dc_left_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* left) {
	int expected_dc, sum = 0;

	for (int i = 0; i < bs; ++i) sum += left[i];
	expected_dc = (sum + (bs >> 1)) / bs;

	for (int r = 0; r < 4; ++r) {
		cuda_vpx_memset16(dst, expected_dc, 4);
		dst += stride;
	}
}

__device__ static void highbd_dc_top_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* above) {
	int expected_dc, sum = 0;

	for (int i = 0; i < bs; ++i) sum += above[i];
	expected_dc = (sum + (bs >> 1)) / bs;

	for (int r = 0; r < 4; ++r) {
		cuda_vpx_memset16(dst, expected_dc, 4);
		dst += stride;
	}
}

__device__ static void highbd_dc_predictor(uint16_t* dst, ptrdiff_t stride, int bs, const uint16_t* above, const uint16_t* left, int bd) {
	int i, r, expected_dc, sum = 0;
	const int count = 2 * bs;
	(void)bd;

	for (i = 0; i < bs; i++) {
		sum += above[i];
		sum += left[i];
	}

	expected_dc = (sum + (count >> 1)) / count;

	for (r = 0; r < 4; r++) {
		cuda_vpx_memset16(dst, expected_dc, 4);
		dst += stride;
	}
}

__device__ static void cuda_build_intra_predictors_high(uint16_t* dst, int stride, PREDICTION_MODE mode, const int bs, int up_available,
	int left_available, int right_available, int x0, int y0, int plane, FrameInformation* fi, int mb_to_bottom_edge,
	int mb_to_right_edge, int ty, int tx) {
	const uint16_t* ref = dst;
	const uint8_t bd = fi->bit_depth;
	int i;

	//sleep
	__shared__ uint16_t left[32 * intraThreadsPerBlockSleep];
	uint16_t* left_col = &left[threadIdx.x * 32];
	DECLARE_ALIGNED(16, uint16_t, above_data[64 + 16]);
	uint16_t* above_row = above_data + 16;

	//traditional
	//__shared__ uint16_t left[32 * intraThreadsPerBlockTr];
	//uint16_t* left_col = &left[threadIdx.x * 32];
	//__shared__ uint16_t above_data[(64 + 16) * intraThreadsPerBlockSleep];
	//uint16_t* above_row = &above_data[threadIdx.x * 80] + 16;

	const uint16_t* const_above_row = above_row;
	int frame_width, frame_height;
	const int need_left = cuda_extend_modes[mode] & NEED_LEFT;
	const int need_above = cuda_extend_modes[mode] & NEED_ABOVE;
	const int need_aboveright = cuda_extend_modes[mode] & NEED_ABOVERIGHT;
	int base = 128 << (bd - 8);
	// 127 127 127 .. 127 127 127 127 127 127
	// 129  A   B  ..  Y   Z
	// 129  C   D  ..  W   X
	// 129  E   F  ..  U   V
	// 129  G   H  ..  S   T   T   T   T   T
	// For 10 bit and 12 bit, 127 and 129 are replaced by base -1 and base + 1.

	// Get current frame pointer, width and height.
	frame_width = (plane == 0) ? fi->y_crop_width : fi->uv_crop_width;
	frame_height = (plane == 0) ? fi->y_crop_height : fi->uv_crop_height;

	// NEED_LEFT
	if (need_left) {
		if (left_available) {
			if (mb_to_bottom_edge < 0) {
				/* slower path if the block needs border extension */
				if (y0 + bs <= frame_height) {
					for (i = 0; i < bs; ++i) left_col[i] = ref[i * stride - 1];
				}
				else {
					const int extend_bottom = frame_height - y0;
					for (i = 0; i < extend_bottom; ++i)
						left_col[i] = ref[i * stride - 1];
					for (; i < bs; ++i)
						left_col[i] = ref[(extend_bottom - 1) * stride - 1];
				}
			}
			else {
				/* faster path if the block does not need extension */
				for (i = 0; i < bs; ++i) left_col[i] = ref[i * stride - 1];
			}
		}
		else {
			cuda_vpx_memset16(left_col, base + 1, bs);
		}
	}

	// NEED_ABOVE
	if (need_above) {
		if (up_available) {
			const uint16_t* above_ref = ref - stride;
			if (mb_to_right_edge < 0) {
				/* slower path if the block needs border extension */
				if (x0 + bs <= frame_width) {
					memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
				}
				else if (x0 <= frame_width) {
					const int r = frame_width - x0;
					memcpy(above_row, above_ref, r * sizeof(above_row[0]));
					cuda_vpx_memset16(above_row + r, above_row[r - 1], x0 + bs - frame_width);
				}
			}
			else {
				/* faster path if the block does not need extension */
				if (bs == 4 && right_available && left_available) {
					const_above_row = above_ref;
				}
				else {
					memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
				}
			}
			above_row[-1] = left_available ? above_ref[-1] : (base + 1);
		}
		else {
			cuda_vpx_memset16(above_row, base - 1, bs);
			above_row[-1] = base - 1;
		}
	}

	// NEED_ABOVERIGHT
	if (need_aboveright) {
		if (up_available) {
			const uint16_t* above_ref = ref - stride;
			if (mb_to_right_edge < 0) {
				/* slower path if the block needs border extension */
				if (x0 + 2 * bs <= frame_width) {
					if (right_available && bs == 4) {
						memcpy(above_row, above_ref, 2 * bs * sizeof(above_row[0]));
					}
					else {
						memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
						cuda_vpx_memset16(above_row + bs, above_row[bs - 1], bs);
					}
				}
				else if (x0 + bs <= frame_width) {
					const int r = frame_width - x0;
					if (right_available && bs == 4) {
						memcpy(above_row, above_ref, r * sizeof(above_row[0]));
						cuda_vpx_memset16(above_row + r, above_row[r - 1],
							x0 + 2 * bs - frame_width);
					}
					else {
						memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
						cuda_vpx_memset16(above_row + bs, above_row[bs - 1], bs);
					}
				}
				else if (x0 <= frame_width) {
					const int r = frame_width - x0;
					memcpy(above_row, above_ref, r * sizeof(above_row[0]));
					cuda_vpx_memset16(above_row + r, above_row[r - 1],
						x0 + 2 * bs - frame_width);
				}
				above_row[-1] = left_available ? above_ref[-1] : (base + 1);
			}
			else {
				/* faster path if the block does not need extension */
				if (bs == 4 && right_available && left_available) {
					const_above_row = above_ref;
				}
				else {
					memcpy(above_row, above_ref, bs * sizeof(above_row[0]));
					if (bs == 4 && right_available)
						memcpy(above_row + bs, above_ref + bs, bs * sizeof(above_row[0]));
					else
						cuda_vpx_memset16(above_row + bs, above_row[bs - 1], bs);
					above_row[-1] = left_available ? above_ref[-1] : (base + 1);
				}
			}
		}
		else {
			cuda_vpx_memset16(above_row, base - 1, bs * 2);
			above_row[-1] = base - 1;
		}
	}

	uint16_t* tdst = dst + ty * stride + tx;
	
	// predict
	if (mode == DC_PRED) {
		if (left_available == 0 && up_available == 0) highbd_dc_128_predictor(tdst, stride, bs, bd);
		else if (left_available == 0 && up_available == 1) highbd_dc_top_predictor(tdst, stride, bs, const_above_row);
		else if (left_available == 1 && up_available == 0) highbd_dc_left_predictor(tdst, stride, bs, left_col);
		else highbd_dc_predictor(tdst, stride, bs, const_above_row, left_col, bd);
	}
	else {
		switch (mode) {
		case V_PRED: highbd_v_predictor(tdst, stride, &const_above_row[tx]); break;
		case H_PRED: highbd_h_predictor(tdst, stride, &left_col[ty]); break;
		case D207_PRED: highbd_d207_predictor(dst, stride, bs, left_col, ty, tx); break;
		case D45_PRED: highbd_d45_predictor(dst, stride, bs, const_above_row, ty, tx); break;
		case D63_PRED: highbd_d63_predictor(dst, stride, bs, const_above_row, ty, tx); break;
		case D117_PRED: highbd_d117_predictor(dst, stride, const_above_row, left_col, ty, tx); break;
		case D135_PRED: highbd_d135_predictor(tdst, stride, bs, const_above_row, left_col, ty, tx); break;
		case D153_PRED: highbd_d153_predictor(dst, stride, const_above_row, left_col, ty, tx); break;
		case TM_PRED: highbd_tm_predictor(tdst, stride, const_above_row, left_col, bd, ty, tx); break;
		default: break;
		}
	}
}

__device__ static void blockSum_4x4(int bd, int stride, uint16_t* dst_buf, tran_high_t* buf)
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			dst_buf[j * stride + i] = intra_highbd_clip_pixel_add(
				dst_buf[j * stride + i], buf[j * stride + i], bd);
		}
	}
}

__device__ static void cuda_intra_predict_and_reconstruct(FrameInformation* fi, uint8_t* alloc, int16_t y0, int16_t x0, int8_t plane, int8_t block_size,
	PREDICTION_MODE mode, int up_available, int left_available, int right_available, tran_high_t* residuals_alloc, int8_t skip,
	int mb_to_bottom_edge, int mb_to_right_edge, int16_t ty, int16_t tx) {
	uint8_t* y_buf = (uint8_t*)yv12_align_addr(alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
	uint8_t* u_buf = (uint8_t*)yv12_align_addr(alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
	uint8_t* v_buf = (uint8_t*)yv12_align_addr(alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);

	uint8_t* buffer = (plane == 0 ? y_buf : (plane == 1 ? u_buf : v_buf));
	int stride = (plane == 0 ? fi->y_stride : fi->uv_stride);

	uint16_t* dst = CONVERT_TO_SHORTPTR(buffer) + y0 * stride + x0;
	cuda_build_intra_predictors_high(dst, stride, mode, block_size, up_available, left_available, right_available, x0, y0, plane, fi,
		mb_to_bottom_edge, mb_to_right_edge, ty, tx);

	if (skip) {
		tran_high_t* y_residuals = (tran_high_t*)yv12_align_addr(residuals_alloc + (fi->border * fi->y_stride) + fi->border, fi->vp9_byte_align);
		tran_high_t* u_residuals = (tran_high_t*)yv12_align_addr(residuals_alloc + fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);
		tran_high_t* v_residuals = (tran_high_t*)yv12_align_addr(residuals_alloc + fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w, fi->vp9_byte_align);

		tran_high_t* residuals_buffer = plane == 0 ? y_residuals : (plane == 1 ? u_residuals : v_residuals);
		residuals_buffer += (y0 + ty) * stride + x0 + tx;
		dst = CONVERT_TO_SHORTPTR(buffer) + (y0 + ty) * stride + x0 + tx;
		
		blockSum_4x4(fi->bit_depth, stride, dst, residuals_buffer);
	}
}





//____________________________________________________________SLEEP_____________________________________________________________________________

__device__ int atomicLoad(int8_t* addr)
{
	volatile int8_t* vaddr = addr; // volatile to bypass cache
	__threadfence(); // for seq_cst loads. Remove for acquire semantics.
	int8_t value = *vaddr;
	return value;
}

__device__ static bool canDecode(int8_t* block_decoded, PREDICTION_MODE mode, int bs, int stride, int8_t up_available,
	int8_t left_available, int8_t right_available)
{
	const int need_left = cuda_extend_modes[mode] & NEED_LEFT;
	const int need_above = cuda_extend_modes[mode] & NEED_ABOVE;
	const int need_aboveright = cuda_extend_modes[mode] & NEED_ABOVERIGHT;

	if (need_left)
	{
		if (left_available)
		{
			for (int i = 0; i < bs; i+=4)
			{
				if (atomicLoad(&block_decoded[i * stride - 4]) == 0) return false;
			}
		}
	}

	if (need_above)
	{
		if (up_available)
		{
			int8_t* bd = block_decoded - 4 * stride;
			if (left_available && atomicLoad(&bd[-4]) == 0) return false;
			for (int i = 0; i < bs; i+=4)
			{
				if (atomicLoad(&bd[i]) == 0) return false;
			}
		}
	}

	if (need_aboveright)
	{
		if (up_available)
		{
			int8_t* bd = block_decoded - 4 * stride;
			if (left_available && atomicLoad(&bd[-4]) == 0) return false;
			if (right_available)
			{
				for (int i = 0; i < 2 * bs; i+=4)
				{
					if (atomicLoad(&bd[i]) == 0) return false;
				}
			}
			else
			{
				for (int i = 0; i < bs; i+=4)
				{
					if (atomicLoad(&bd[i]) == 0) return false;
				}
			}
		}
	}

	return true;
}

__global__ static void intra_sleep(uint8_t* alloc, FrameInformation* fi, const int super_size, VP9Block* block_settings,
	tran_high_t* residuals, int8_t* block_decoded)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	uint8_t* converted_alloc = alloc;

#if CONFIG_VP9_HIGHBITDEPTH
	converted_alloc = CONVERT_TO_BYTEPTR(converted_alloc);

	if (i < super_size) {
		int16_t x = block_settings[i].x;
		int16_t y = block_settings[i].y;
		int16_t tx = block_settings[i].tx;
		int16_t ty = block_settings[i].ty;
		PREDICTION_MODE mode = block_settings[i].mode;
		int8_t block_size = block_settings[i].block_size;
		int8_t plane = block_settings[i].plane;
		int stride = (plane == 0) ? fi->y_stride : fi->uv_stride;
		int8_t up_available = block_settings[i].have_top;
		int8_t left_available = block_settings[i].have_left;
		int8_t right_available = block_settings[i].have_right;

		size_t y_border = (fi->border * fi->y_stride) + fi->border;
		size_t u_border = fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w;
		size_t v_border = fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w;

		int8_t* bd = (plane == 0) ? (block_decoded + y_border) : ((plane == 1) ? (block_decoded + u_border) : (block_decoded + v_border));
		bd += y * stride + x;

		while (!canDecode(bd, mode, block_size, stride, up_available, left_available, right_available))
		{
			__nanosleep(ns);
		}

		cuda_intra_predict_and_reconstruct(fi, converted_alloc, y, x, plane, block_size, mode, up_available, left_available,
			right_available, residuals, block_settings[i].skip, block_settings[i].mb_to_bottom_edge, block_settings[i].mb_to_right_edge,
			ty, tx);

		bd[ty * stride + tx] = 1;
	}
#endif
}

__global__ static void set_decoded(FrameInformation* fi, VP9Block* block_settings, int8_t* block_decoded, const int super_size)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < super_size)
	{
		int x = block_settings[i].x;
		int y = block_settings[i].y;
		int16_t tx = block_settings[i].tx;
		int16_t ty = block_settings[i].ty;
		int8_t plane = block_settings[i].plane;
		int stride = (plane == 0) ? fi->y_stride : fi->uv_stride;

		size_t y_border = (fi->border * fi->y_stride) + fi->border;
		size_t u_border = fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w;
		size_t v_border = fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w;

		int8_t* bd = (plane == 0) ? (block_decoded + y_border) : ((plane == 1) ? (block_decoded + u_border) : (block_decoded + v_border));
		
		bd[(y + ty) * stride + x + tx] = 0;
	}
}

static MODE_INFO* set_offsets(VP9_COMMON* const cm, MACROBLOCKD* const xd, BLOCK_SIZE bsize, int mi_row, int mi_col, int bw,
	int bh, int x_mis, int y_mis, int bwl, int bhl) {
	const int offset = mi_row * cm->mi_stride + mi_col;
	int x, y;
	const TileInfo* const tile = &xd->tile;

	xd->mi = cm->mi_grid_visible + offset;
	xd->mi[0] = &cm->mi[offset];
	// TODO(slavarnway): Generate sb_type based on bwl and bhl, instead of
	// passing bsize from decode_partition().
	xd->mi[0]->sb_type = bsize;
	for (y = 0; y < y_mis; ++y)
		for (x = !y; x < x_mis; ++x) {
			xd->mi[y * cm->mi_stride + x] = xd->mi[0];
		}

	int i;
	for (i = 0; i < MAX_MB_PLANE; i++) {
		xd->plane[i].n4_w = (bw << 1) >> xd->plane[i].subsampling_x;
		xd->plane[i].n4_h = (bh << 1) >> xd->plane[i].subsampling_y;
		xd->plane[i].n4_wl = bwl - xd->plane[i].subsampling_x;
		xd->plane[i].n4_hl = bhl - xd->plane[i].subsampling_y;
	}

	// Are edges available for intra prediction?
	xd->above_mi = (mi_row != 0) ? xd->mi[-xd->mi_stride] : NULL;
	xd->left_mi = (mi_col > tile->mi_col_start) ? xd->mi[-1] : NULL;

	return xd->mi[0];
}

__host__ int createBuffers(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols,
	VP9Block_t* host_block_settings, FrameInformation* host_fi, frameBuf* frameBuffer)
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

	int stride;

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
				for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
					for (int i = 0; i < *strt_size_for_mb; ++i) {
						if (!is_inter_block(strt_mi_buf.mi[0])) {
							const int bh = 1 << (*strt_mi_buf.bhl - 1);
							const int bw = 1 << (*strt_mi_buf.bwl - 1);
							const int x_mis = VPXMIN(bw, cm->mi_cols - *strt_mi_buf.mi_col);
							const int y_mis = VPXMIN(bh, cm->mi_rows - *strt_mi_buf.mi_row);
							MACROBLOCKD* const xd = &tile_data->xd;
							set_offsets(cm, xd, strt_mi_buf.mi[0]->sb_type, *strt_mi_buf.mi_row, *strt_mi_buf.mi_col, bw, bh, x_mis, y_mis, *strt_mi_buf.bwl, *strt_mi_buf.bhl);
							int plane;
							for (plane = 0; plane < MAX_MB_PLANE; ++plane)
							{
								const struct macroblockd_plane* const pd = &xd->plane[plane];
								const TX_SIZE tx_size = plane ? get_uv_tx_size(strt_mi_buf.mi[0], pd) : strt_mi_buf.mi[0]->tx_size;
								int t_subsampling_x = pd->subsampling_x;
								const int step = (1 << tx_size);
								const int num_4x4_w = pd->n4_w;
								const int num_4x4_h = pd->n4_h;
								int mb_to_right_edge = ((cm->mi_cols - bw - mi_col) * MI_SIZE) * 8;
								int mb_to_bottom_edge = ((cm->mi_rows - bh - mi_row) * MI_SIZE) * 8;
								PREDICTION_MODE mode = (plane == 0) ? strt_mi_buf.mi[0]->mode : strt_mi_buf.mi[0]->uv_mode;
								stride = (plane == 0 ? host_fi->y_stride : host_fi->uv_stride);
								const int max_blocks_wide = num_4x4_w + (mb_to_right_edge >= 0 ? 0 : mb_to_right_edge >> (5 + pd->subsampling_x));
								const int max_blocks_high = num_4x4_h + (mb_to_bottom_edge >= 0 ? 0 : mb_to_bottom_edge >> (5 + pd->subsampling_y));

								xd->max_blocks_wide = mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
								xd->max_blocks_high = mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

								int strt_by = (*strt_mi_buf.mi_row * MI_SIZE * 8) >> (3 + t_subsampling_x);
								int strt_bx = (*strt_mi_buf.mi_col * MI_SIZE * 8) >> (3 + t_subsampling_x);
								int* eob_buf = frameBuffer->plane_eob[plane] + strt_by * stride + strt_bx;

								for (int row = 0; row < max_blocks_high; row += step)
									for (int col = 0; col < max_blocks_wide; col += step) {
										if (strt_mi_buf.mi[0]->sb_type < BLOCK_8X8)
											if (plane == 0) mode = strt_mi_buf.mi[0]->bmi[(row << 1) + col].as_mode;

										const int bw = (1 << pd->n4_wl);
										const int txw = (1 << tx_size);

										uint16_t dx = strt_bx + 4 * col;
										uint16_t dy = strt_by + 4 * row;

										uint8_t have_left = col || (xd->left_mi != NULL);
										uint8_t have_right = (col + txw) < bw;
										uint8_t have_top = row || (xd->above_mi != NULL);
										int8_t bs = 4 << tx_size;

										uint8_t skip;
										if (!strt_mi_buf.mi[0]->skip)
										{
											if (eob_buf[4 * row * stride + 4 * col] > 0) skip = 1;
											else skip = 0;
										}
										else skip = 0;

										for (int y = 0; y < bs; y += 4) {
											for (int x = 0; x < bs; x += 4) {
												host_block_settings[super_size].tx = x;
												host_block_settings[super_size].ty = y;
												host_block_settings[super_size].x = dx;
												host_block_settings[super_size].y = dy;
												host_block_settings[super_size].plane = plane;
												host_block_settings[super_size].mode = mode;
												host_block_settings[super_size].mb_to_bottom_edge = mb_to_bottom_edge;
												host_block_settings[super_size].mb_to_right_edge = mb_to_right_edge;
												host_block_settings[super_size].have_left = have_left;
												host_block_settings[super_size].have_right = have_right;
												host_block_settings[super_size].have_top = have_top;
												host_block_settings[super_size].block_size = bs;
												host_block_settings[super_size].skip = skip;
												++super_size;
											}
										}
									}
							}
						}

						++strt_mi_buf.mi_col;
						++strt_mi_buf.mi;
						++strt_mi_buf.mi_row;
						++strt_mi_buf.bhl;
						++strt_mi_buf.bwl;
					}
					++strt_size_for_mb;
				}
			}
		}
	}

	return super_size;
}

__host__ int cuda_intra_prediction_sl(double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
	VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols, frameBuf* frameBuffer)
{
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);

	uint8_t* cd_alloc;
	tran_high_t* cd_residuals;
	FrameInformation* cd_fi, * host_fi;
	VP9Block* block_settings, * host_block_settings;
	int8_t* block_decoded;

	size_t sz = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.frame_size;
	uint8_t* alloc = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.buffer_alloc;

	cudaHostAlloc((void**)&host_block_settings, sz * sizeof(VP9Block), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_fi, sizeof(FrameInformation), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	int super_size = createBuffers(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, host_block_settings, host_fi, frameBuffer);

	cudaMallocAsync((void**)&cd_alloc, host_fi->size, stream1);
	cudaMallocAsync((void**)&block_decoded, host_fi->size, stream2);
	cudaMallocAsync((void**)&cd_residuals, host_fi->size * sizeof(tran_high_t), stream3);
	cudaDeviceSynchronize();

	clock_t copy_begin = clock();
	//copy
	{
		cudaHostGetDevicePointer((void**)&cd_fi, (void*)host_fi, 0);
		cudaHostGetDevicePointer((void**)&block_settings, (void*)host_block_settings, 0);
		cudaMemcpyAsync(cd_alloc, alloc, host_fi->size, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(cd_residuals, frameBuffer->residuals, host_fi->size * sizeof(tran_high_t), cudaMemcpyHostToDevice, stream2);
		cudaMemsetAsync((void*)block_decoded, 1, host_fi->size, stream3);
		cudaDeviceSynchronize();
	}
	clock_t copy_end = clock();
	*gpu_copy = (double)(copy_end - copy_begin) / CLOCKS_PER_SEC;

	float elapsed = 0;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int blocksPerGrid = MAX(1, (super_size + intraThreadsPerBlockSleep - 1) / intraThreadsPerBlockSleep);
	set_decoded <<<blocksPerGrid, intraThreadsPerBlockSleep >>> (cd_fi, block_settings, block_decoded, super_size);
	cudaDeviceSynchronize();
	
	intra_sleep <<<blocksPerGrid, intraThreadsPerBlockSleep >>> (cd_alloc, cd_fi, super_size, block_settings, cd_residuals, block_decoded);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*gpu_run = ((double)elapsed) / 1000;

	cudaMemcpy(alloc, cd_alloc, host_fi->size, cudaMemcpyDeviceToHost);

	//free
	{
		cudaFreeHost(host_block_settings);
		cudaFreeAsync((void*)block_decoded, stream1);
		cudaFreeAsync(cd_alloc, stream2);
		cudaFreeAsync(cd_residuals, stream3);
		cudaFreeHost(host_fi);
	}

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);

	return 0;
}





//___________________________________________________________WITHOUT SLEEP_________________________________________________________________

__global__ static void intra_traditional(uint8_t* alloc, FrameInformation* fi, const int super_size, VP9Block* block_settings,
	tran_high_t* residuals, int startId)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	uint8_t* converted_alloc = alloc;

#if CONFIG_VP9_HIGHBITDEPTH
	converted_alloc = CONVERT_TO_BYTEPTR(converted_alloc);

	if (i < super_size) {
		i += startId;
		int16_t x = block_settings[i].x;
		int16_t y = block_settings[i].y;
		int16_t tx = block_settings[i].tx;
		int16_t ty = block_settings[i].ty;
		PREDICTION_MODE mode = block_settings[i].mode;
		int8_t block_size = block_settings[i].block_size;
		int8_t plane = block_settings[i].plane;
		int8_t up_available = block_settings[i].have_top;
		int8_t left_available = block_settings[i].have_left;
		int8_t right_available = block_settings[i].have_right;

		cuda_intra_predict_and_reconstruct(fi, converted_alloc, y, x, plane, block_size, mode, up_available, left_available,
			right_available, residuals, block_settings[i].skip, block_settings[i].mb_to_bottom_edge, block_settings[i].mb_to_right_edge,
			ty, tx);

	}
#endif
}

__host__ void setFilterd(int8_t* block_decoded, VP9Block_t* block_settings, FrameInformation* fi, int iterationSize)
{
	for(int i = 0; i < iterationSize; ++i)
	{
		int x = block_settings[i].x;
		int y = block_settings[i].y;
		int16_t tx = block_settings[i].tx;
		int16_t ty = block_settings[i].ty;
		int8_t plane = block_settings[i].plane;
		int stride = (plane == 0) ? fi->y_stride : fi->uv_stride;

		size_t y_border = (fi->border * fi->y_stride) + fi->border;
		size_t u_border = fi->yplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w;
		size_t v_border = fi->yplane_size + fi->uvplane_size + (fi->uv_border_h * fi->uv_stride) + fi->uv_border_w;

		int8_t* bd = (plane == 0) ? (block_decoded + y_border) : ((plane == 1) ? (block_decoded + u_border) : (block_decoded + v_border));

		bd += (y + ty) * stride + x + tx;
		
		for (int j = 0; j < 4; ++j)
		{
			for (int k = 0; k < 4; ++k) bd[k * stride + j] = 1;
		}
	}
}

__host__ static bool canDecodeHost(int8_t* block_decoded, PREDICTION_MODE mode, int bs, int stride, int8_t up_available,
	int8_t left_available, int8_t right_available)
{
	uint8_t cpu_extend_modes[INTRA_MODES] = {
		NEED_ABOVE | NEED_LEFT,  // DC
		NEED_ABOVE,              // V
		NEED_LEFT,               // H
		NEED_ABOVERIGHT,         // D45
		NEED_LEFT | NEED_ABOVE,  // D135
		NEED_LEFT | NEED_ABOVE,  // D117
		NEED_LEFT | NEED_ABOVE,  // D153
		NEED_LEFT,               // D207
		NEED_ABOVERIGHT,         // D63
		NEED_LEFT | NEED_ABOVE,  // TM
	};
	
	const int need_left = cpu_extend_modes[mode] & NEED_LEFT;
	const int need_above = cpu_extend_modes[mode] & NEED_ABOVE;
	const int need_aboveright = cpu_extend_modes[mode] & NEED_ABOVERIGHT;

	if (need_left)
	{
		if (left_available)
		{
			for (int i = 0; i < bs; ++i)
			{
				if (block_decoded[i * stride - 1] == 0) return false;
			}
		}
	}

	if (need_above)
	{
		if (up_available)
		{
			int8_t* bd = block_decoded - stride;
			if (left_available && bd[-1] == 0) return false;
			for (int i = 0; i < bs; ++i)
			{
				if (bd[i] == 0) return false;
			}
		}
	}

	if (need_aboveright)
	{
		if (up_available)
		{
			int8_t* bd = block_decoded - stride;
			if (left_available && bd[-1] == 0) return false;
			if (right_available)
			{
				for (int i = 0; i < 2 * bs; ++i)
				{
					if (bd[i] == 0) return false;
				}
			}
			else
			{
				for (int i = 0; i < bs; ++i)
				{
					if (bd[i] == 0) return false;
				}
			}
		}
	}

	return true;
}

__host__ static unsigned int globalCount(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, 
	int tile_rows, int tile_cols, int8_t* block_decoded, FrameInformation* host_fi)
{
	ModeInfoBuf strt_mi_buf;
	strt_mi_buf.bhl = MiBuf->bhl;
	strt_mi_buf.bwl = MiBuf->bwl;
	strt_mi_buf.mi_col = MiBuf->mi_col;
	strt_mi_buf.mi_row = MiBuf->mi_row;
	strt_mi_buf.mi = MiBuf->mi;
	int* strt_size_for_mb = size_for_mb;
	
	TileWorkerData* tile_data;
	int tile_row, mi_col, tile_col, mi_row, stride;
	unsigned int super_size = 0;

	size_t y_border = (host_fi->border * host_fi->y_stride) + host_fi->border;
	size_t u_border = host_fi->yplane_size + (host_fi->uv_border_h * host_fi->uv_stride) + host_fi->uv_border_w;
	size_t v_border = host_fi->yplane_size + host_fi->uvplane_size + (host_fi->uv_border_h * host_fi->uv_stride) + host_fi->uv_border_w;


	for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
		TileInfo tile;
		vp9_tile_set_row(&tile, cm, tile_row);
		for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
			for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
				const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
				tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
				vp9_tile_set_col(&tile, cm, col);
				for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
					for (int i = 0; i < *strt_size_for_mb; ++i) {
						if (!is_inter_block(strt_mi_buf.mi[0])) {
							const int bh = 1 << (*strt_mi_buf.bhl - 1);
							const int bw = 1 << (*strt_mi_buf.bwl - 1);
							const int x_mis = VPXMIN(bw, cm->mi_cols - *strt_mi_buf.mi_col);
							const int y_mis = VPXMIN(bh, cm->mi_rows - *strt_mi_buf.mi_row);
							MACROBLOCKD* const xd = &tile_data->xd;
							set_offsets(cm, xd, strt_mi_buf.mi[0]->sb_type, *strt_mi_buf.mi_row, *strt_mi_buf.mi_col, 
								bw, bh, x_mis, y_mis, *strt_mi_buf.bwl, *strt_mi_buf.bhl);
							int plane;
							for (plane = 0; plane < MAX_MB_PLANE; ++plane)
							{
								const struct macroblockd_plane* const pd = &xd->plane[plane];
								const TX_SIZE tx_size = plane ? get_uv_tx_size(strt_mi_buf.mi[0], pd) : strt_mi_buf.mi[0]->tx_size;
								int t_subsampling_x = pd->subsampling_x;
								const int step = (1 << tx_size);
								const int num_4x4_w = pd->n4_w;
								const int num_4x4_h = pd->n4_h;
								int mb_to_right_edge = ((cm->mi_cols - bw - mi_col) * MI_SIZE) * 8;
								int mb_to_bottom_edge = ((cm->mi_rows - bh - mi_row) * MI_SIZE) * 8;
								stride = (plane == 0 ? host_fi->y_stride : host_fi->uv_stride);
								const int max_blocks_wide = num_4x4_w + (mb_to_right_edge >= 0 ? 0 : mb_to_right_edge >> (5 + pd->subsampling_x));
								const int max_blocks_high = num_4x4_h + (mb_to_bottom_edge >= 0 ? 0 : mb_to_bottom_edge >> (5 + pd->subsampling_y));

								xd->max_blocks_wide = mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
								xd->max_blocks_high = mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

								int strt_by = (*strt_mi_buf.mi_row * MI_SIZE * 8) >> (3 + t_subsampling_x);
								int strt_bx = (*strt_mi_buf.mi_col * MI_SIZE * 8) >> (3 + t_subsampling_x);
								
								int8_t* bd0 = (plane == 0) ? (block_decoded + y_border) : ((plane == 1) ? (block_decoded + u_border) : (block_decoded + v_border));

								for (int row = 0; row < max_blocks_high; row += step)
									for (int col = 0; col < max_blocks_wide; col += step) {
										uint16_t dx = strt_bx + 4 * col;
										uint16_t dy = strt_by + 4 * row;
										int8_t bs = 4 << tx_size;
										int8_t* bd = bd0 + dy * stride + dx;
										
										for(int k = 0; k < bs; ++k)
										{
											for(int j = 0; j < bs; ++j)
											{
												bd[k * stride + j] = 0;
											}
										}
										
										super_size += bs * bs / 16;
									}
							}
						}

						++strt_mi_buf.mi_col;
						++strt_mi_buf.mi;
						++strt_mi_buf.mi_row;
						++strt_mi_buf.bhl;
						++strt_mi_buf.bwl;
					}
					++strt_size_for_mb;
				}
			}
		}
	}

	return super_size;
}

__host__ static unsigned int frameAnalyz(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols,
	VP9Block_t* host_block_settings, FrameInformation* host_fi, frameBuf* frameBuffer, int8_t* block_decoded)
{
	ModeInfoBuf strt_mi_buf;
	strt_mi_buf.bhl = MiBuf->bhl;
	strt_mi_buf.bwl = MiBuf->bwl;
	strt_mi_buf.mi_col = MiBuf->mi_col;
	strt_mi_buf.mi_row = MiBuf->mi_row;
	strt_mi_buf.mi = MiBuf->mi;
	int* strt_size_for_mb = size_for_mb;
	
	int stride;
	TileWorkerData* tile_data;
	int tile_row, mi_col, tile_col, mi_row;
	unsigned int super_size = 0;

	size_t y_border = (host_fi->border * host_fi->y_stride) + host_fi->border;
	size_t u_border = host_fi->yplane_size + (host_fi->uv_border_h * host_fi->uv_stride) + host_fi->uv_border_w;
	size_t v_border = host_fi->yplane_size + host_fi->uvplane_size + (host_fi->uv_border_h * host_fi->uv_stride) + host_fi->uv_border_w;

	
	for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
		TileInfo tile;
		vp9_tile_set_row(&tile, cm, tile_row);
		for (mi_row = tile.mi_row_start; mi_row < tile.mi_row_end; mi_row += MI_BLOCK_SIZE) {
			for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
				const int col = pbi->inv_tile_order ? tile_cols - tile_col - 1 : tile_col;
				tile_data = pbi->tile_worker_data + tile_cols * tile_row + col;
				vp9_tile_set_col(&tile, cm, col);
				for (mi_col = tile.mi_col_start; mi_col < tile.mi_col_end; mi_col += MI_BLOCK_SIZE) {
					for (int i = 0; i < *strt_size_for_mb; ++i) {
						if (!is_inter_block(strt_mi_buf.mi[0])) {
							const int bh = 1 << (*strt_mi_buf.bhl - 1);
							const int bw = 1 << (*strt_mi_buf.bwl - 1);
							const int x_mis = VPXMIN(bw, cm->mi_cols - *strt_mi_buf.mi_col);
							const int y_mis = VPXMIN(bh, cm->mi_rows - *strt_mi_buf.mi_row);
							MACROBLOCKD* const xd = &tile_data->xd;
							set_offsets(cm, xd, strt_mi_buf.mi[0]->sb_type, *strt_mi_buf.mi_row, *strt_mi_buf.mi_col, 
								bw, bh, x_mis, y_mis, *strt_mi_buf.bwl, *strt_mi_buf.bhl);
							int plane;
							for (plane = 0; plane < MAX_MB_PLANE; ++plane)
							{
								const struct macroblockd_plane* const pd = &xd->plane[plane];
								const TX_SIZE tx_size = plane ? get_uv_tx_size(strt_mi_buf.mi[0], pd) : strt_mi_buf.mi[0]->tx_size;
								int t_subsampling_x = pd->subsampling_x;
								const int step = (1 << tx_size);
								const int num_4x4_w = pd->n4_w;
								const int num_4x4_h = pd->n4_h;
								int mb_to_right_edge = ((cm->mi_cols - bw - mi_col) * MI_SIZE) * 8;
								int mb_to_bottom_edge = ((cm->mi_rows - bh - mi_row) * MI_SIZE) * 8;
								PREDICTION_MODE mode = (plane == 0) ? strt_mi_buf.mi[0]->mode : strt_mi_buf.mi[0]->uv_mode;
								stride = (plane == 0 ? host_fi->y_stride : host_fi->uv_stride);
								const int max_blocks_wide = num_4x4_w + (mb_to_right_edge >= 0 ? 0 : mb_to_right_edge >> (5 + pd->subsampling_x));
								const int max_blocks_high = num_4x4_h + (mb_to_bottom_edge >= 0 ? 0 : mb_to_bottom_edge >> (5 + pd->subsampling_y));

								xd->max_blocks_wide = mb_to_right_edge >= 0 ? 0 : max_blocks_wide;
								xd->max_blocks_high = mb_to_bottom_edge >= 0 ? 0 : max_blocks_high;

								int strt_by = (*strt_mi_buf.mi_row * MI_SIZE * 8) >> (3 + t_subsampling_x);
								int strt_bx = (*strt_mi_buf.mi_col * MI_SIZE * 8) >> (3 + t_subsampling_x);
								int* eob_buf = frameBuffer->plane_eob[plane] + strt_by * stride + strt_bx;

								int8_t* bd0 = (plane == 0) ? (block_decoded + y_border) : ((plane == 1) ? (block_decoded + u_border) : (block_decoded + v_border));

								for (int row = 0; row < max_blocks_high; row += step)
									for (int col = 0; col < max_blocks_wide; col += step) {
										if (strt_mi_buf.mi[0]->sb_type < BLOCK_8X8)
											if (plane == 0) mode = strt_mi_buf.mi[0]->bmi[(row << 1) + col].as_mode;
										
										const int bw = (1 << pd->n4_wl);
										const int txw = (1 << tx_size);

										uint16_t dx = strt_bx + 4 * col;
										uint16_t dy = strt_by + 4 * row;
										int8_t* bd = bd0 + dy * stride + dx;
										
										uint8_t have_left = col || (xd->left_mi != NULL);
										uint8_t have_right = (col + txw) < bw;
										uint8_t have_top = row || (xd->above_mi != NULL);
										int8_t bs = 4 << tx_size;

										if (bd[0] == 0)
										{
											if (canDecodeHost(bd, mode, bs, stride, have_top, have_left, have_right))
											{
												uint8_t skip = 0;
												if (!strt_mi_buf.mi[0]->skip && eob_buf[4 * row * stride + 4 * col] > 0)
													skip = 1;
												
												for (int y = 0; y < bs; y += 4) {
													for (int x = 0; x < bs; x += 4) {
														host_block_settings[super_size].tx = x;
														host_block_settings[super_size].ty = y;
														host_block_settings[super_size].x = dx;
														host_block_settings[super_size].y = dy;
														host_block_settings[super_size].plane = plane;
														host_block_settings[super_size].mode = mode;
														host_block_settings[super_size].mb_to_bottom_edge = mb_to_bottom_edge;
														host_block_settings[super_size].mb_to_right_edge = mb_to_right_edge;
														host_block_settings[super_size].have_left = have_left;
														host_block_settings[super_size].have_right = have_right;
														host_block_settings[super_size].have_top = have_top;
														host_block_settings[super_size].block_size = bs;
														host_block_settings[super_size].skip = skip;
														++super_size;
													}
												}
												
											}
										}
									}
							}
						}

						++strt_mi_buf.mi_col;
						++strt_mi_buf.mi;
						++strt_mi_buf.mi_row;
						++strt_mi_buf.bhl;
						++strt_mi_buf.bwl;
					}
					++strt_size_for_mb;
				}
			}
		}
	}

	return super_size;
}

__host__ unsigned int createBuffersTr(int* size_for_mb, ModeInfoBuf* MiBuf, VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols,
	VP9Block_t* host_block_settings, FrameInformation* host_fi, frameBuf* frameBuffer, uint16_t* iteratorBuffer)
{
	YV12_BUFFER_CONFIG* src = &cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf;
	VP9Block_t* block_set = host_block_settings;
	int8_t* block_decoded = (int8_t*)malloc(src->frame_size);
	memset(block_decoded, 1, src->frame_size);
	
	const int uv_border_h = src->border >> src->subsampling_y;
	const int uv_border_w = src->border >> src->subsampling_x;

	const int byte_alignment = cm->byte_alignment;
	const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
	const uint64_t yplane_size = (src->y_height + 2 * src->border) * (uint64_t)src->y_stride + byte_alignment;
	const uint64_t uvplane_size = (src->uv_height + 2 * uv_border_h) * (uint64_t)src->uv_stride + byte_alignment;

	//set host_fi and fi
	{
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
	}

	unsigned int globalCnt = globalCount(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, block_decoded, host_fi);
	
	int i = 0;
	int j = 0;
	while (i < globalCnt)
	{
		const unsigned int iterationSize = frameAnalyz(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, block_set, host_fi, frameBuffer, block_decoded);
		setFilterd(block_decoded, block_set, host_fi, iterationSize);
		block_set += iterationSize;
		iteratorBuffer[j] = iterationSize;
		i += iterationSize;
		++j;
	}

	free(block_decoded);
	
	return globalCnt;
}

__host__ int cuda_intra_prediction(double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
	VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols, frameBuf* frameBuffer)
{
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);

	uint8_t* cd_alloc;
	tran_high_t* cd_residuals;
	FrameInformation* cd_fi, * host_fi;
	VP9Block* block_settings, * host_block_settings;
	uint16_t* iteratorBuffer;

	size_t sz = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.frame_size;
	uint8_t* alloc = cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf.buffer_alloc;

	cudaHostAlloc((void**)&host_block_settings, sz * sizeof(VP9Block), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&host_fi, sizeof(FrameInformation), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	iteratorBuffer = (uint16_t*)malloc(sz * sizeof(uint16_t));
	
	unsigned int super_size = createBuffersTr(size_for_mb, MiBuf, cm, pbi, tile_rows, tile_cols, host_block_settings, host_fi, frameBuffer, iteratorBuffer);

	cudaMallocAsync((void**)&cd_alloc, sz, stream1);
	cudaMallocAsync((void**)&cd_residuals, sz * sizeof(tran_high_t), stream2);

	clock_t copy_begin = clock();
	//copy
	{
		cudaHostGetDevicePointer((void**)&cd_fi, (void*)host_fi, 0);
		cudaHostGetDevicePointer((void**)&block_settings, (void*)host_block_settings, 0);
		cudaMemcpyAsync(cd_alloc, alloc, sz, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(cd_residuals, frameBuffer->residuals, sz * sizeof(tran_high_t), cudaMemcpyHostToDevice, stream2);
		//cudaDeviceSynchronize();
	}
	clock_t copy_end = clock();
	*gpu_copy = (double)(copy_end - copy_begin) / CLOCKS_PER_SEC;

	float elapsed = 0;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	int i = 0;
	int j = 0;
	while (i < super_size) {
		int blocksPerGrid = MAX(1, (iteratorBuffer[j] + intraThreadsPerBlockTr - 1) / intraThreadsPerBlockTr);
		intra_traditional <<<blocksPerGrid, intraThreadsPerBlockTr>>> (cd_alloc, cd_fi, iteratorBuffer[j], block_settings, cd_residuals, i);
		i += iteratorBuffer[j];
		++j;
	}
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	*gpu_run = ((double)elapsed) / 1000;

	cudaMemcpy(alloc, cd_alloc, sz, cudaMemcpyDeviceToHost);


	//free
	{
		cudaFreeHost(host_block_settings);
		free(iteratorBuffer);
		cudaFreeAsync(cd_alloc, stream1);
		cudaFreeAsync(cd_residuals, stream2);
		cudaFreeHost(host_fi);
	}

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);

	return 0;
}