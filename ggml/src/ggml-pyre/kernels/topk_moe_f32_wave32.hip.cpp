#include <hip/hip_runtime.h>

#include "topk_moe_f32_common.hip.inc"

#define PYRE_TOPK_MOE_WAVE_KERNEL pyre_topk_moe_f32_wave32
#define PYRE_TOPK_MOE_WAVE_SIZE 32
#include "topk_moe_f32_wave.inc"
