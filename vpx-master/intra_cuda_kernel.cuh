
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "vp9/decoder/vp9_decoder.h"
#include "buffers_struct.h"

#define yv12_align_addr(addr, align) \
  (void *)(((size_t)(addr) + ((align)-1)) & (size_t) - (align))

#define MAX(a, b) ((a > b) ? a : b)

const int intraThreadsPerBlockTr = 64;
const int intraThreadsPerBlockSleep = 256;
const unsigned int ns = 100;

#define DST(x, y) dst[(x) + (y)*stride]
#define AVG3(a, b, c) (((a) + 2 * (b) + (c) + 2) >> 2)
#define AVG2(a, b) (((a) + (b) + 1) >> 1)

enum {
	NEED_LEFT = 1 << 1,
	NEED_ABOVE = 1 << 2,
	NEED_ABOVERIGHT = 1 << 3,
};

__constant__ uint8_t cuda_extend_modes[INTRA_MODES] = {
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

typedef struct VP9Block_t {
    uint16_t x;
    uint16_t y;
	uint16_t tx;
	uint16_t ty;
    int8_t plane;
    int8_t block_size;
    int8_t have_top;
    int8_t have_left;
    int8_t have_right;
    uint8_t mb_to_bottom_edge;
    uint8_t mb_to_right_edge;
    uint8_t skip;
    PREDICTION_MODE mode;
} VP9Block;

__host__ int cuda_intra_prediction(double* gpu_copy, double* gpu_run, int* size_for_mb, ModeInfoBuf* MiBuf,
    VP9_COMMON* cm, VP9Decoder* pbi, int tile_rows, int tile_cols, frameBuf* frameBuffer);