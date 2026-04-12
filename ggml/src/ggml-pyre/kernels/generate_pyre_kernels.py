#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import sys
import tempfile


KERNELS = [
    {
        "name": "pyre_rms_norm_f32",
        "source": "rms_norm.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "pyre_rms_norm_mul_f32",
        "source": "rms_norm_mul_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 144,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "pyre_rms_norm_mul_rope_f32",
        "source": "rms_norm_mul_rope_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 184,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "pyre_rms_norm_mul_rope_set_rows_f32_f16",
        "source": "rms_norm_mul_rope_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 256,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "pyre_add_rms_norm_mul_f32_broadcast",
        "source": "add_rms_norm_mul_f32_broadcast.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 200,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "pyre_add_f32",
        "source": "add_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_add_f32_broadcast",
        "source": "add_f32_broadcast.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_add_add_f32_broadcast",
        "source": "add_add_f32_broadcast.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 144,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_add8_f32",
        "source": "add8_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 9,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_f32",
        "source": "mul_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_f32_broadcast",
        "source": "mul_f32_broadcast.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_div_f32_broadcast",
        "source": "div_f32_broadcast.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_scale_f32",
        "source": "scale_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 16,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_set_rows_f32_f32",
        "source": "set_rows_f32_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_set_rows_f32_f16",
        "source": "set_rows_f32_f16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_set_rows_f32_q8_0",
        "source": "set_rows_f32_q8_0.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_set_rows_f32_q4_0",
        "source": "set_rows_f32_q4_0.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_silu_f32",
        "source": "unary_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_sigmoid_f32",
        "source": "unary_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_sigmoid_mul_f32_strided",
        "source": "sigmoid_mul_f32_strided.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_softplus_f32",
        "source": "unary_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_swiglu_f32",
        "source": "unary_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_sum_rows_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_l2_norm_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_clamp_f32",
        "source": "clamp_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 16,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_get_rows_f32",
        "source": "get_rows_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_get_rows_f32_nr1",
        "source": "get_rows_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_get_rows_q5_k_f32",
        "source": "get_rows_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_concat_f32",
        "source": "concat_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 72,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_copy_strided_f32",
        "source": "copy_strided_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 64,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_copy_f32_f16",
        "source": "copy_f32_f16.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_soft_max_f32",
        "source": "soft_max_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_soft_max_f32_mask",
        "source": "soft_max_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_flash_attn_ext_f32_f16_decode",
        "source": "flash_attn_ext_f32_f16_decode.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_flash_attn_ext_f32_bf16_decode",
        "source": "flash_attn_ext_f32_bf16_decode.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_flash_attn_ext_f32_f32_decode",
        "source": "flash_attn_ext_f32_f32_decode.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_flash_attn_ext_f32_q8_0_decode",
        "source": "flash_attn_ext_f32_q8_0_decode.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_flash_attn_ext_f32_q4_0_decode",
        "source": "flash_attn_ext_f32_q4_0_decode.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_flash_attn_ext_f32_q8_0_q4_0_decode",
        "source": "flash_attn_ext_f32_q8_0_q4_0_decode.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_argsort_f32_i32",
        "source": "argsort_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_rope_f32",
        "source": "rope_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 120,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_rope_set_rows_f32_f16",
        "source": "rope_set_rows_f32_f16.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 192,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_topk_moe_f32",
        "source": "topk_moe_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 64,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_topk_moe_f32_subgroup",
        "source": "topk_moe_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 64,
        "workgroup_size": (64, 4, 1),
    },
    {
        "name": "pyre_ssm_conv_f32",
        "source": "ssm_conv_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_ssm_conv_update_f32",
        "source": "ssm_conv_update_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_gated_delta_net_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 8,
        "constants_size": 208,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_gated_delta_net_s128_cluster16_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 8,
        "constants_size": 208,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f16_f32",
        "source": "mul_mat_vec_f16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f16_batched_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f16_batched_cols1_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f16_batched_cols4_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f16_batched_cols8_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_wg128_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_wg64_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_cols1_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_cols4_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_swiglu_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_swiglu_wg128_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_swiglu_wg64_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_swiglu_cols1_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_swiglu_cols4_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_bf16_set_rows_f16",
        "source": "mul_mat_vec_bf16_set_rows.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 40,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f32_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f32_batched_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_f32_batched_cols1_ne2_1_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_wg128_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_row4_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_row8_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_clear_u32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 1,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_compact_moe_routes_i32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 48,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_grouped_row4_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_q8_1_f32",
        "source": "mul_mat_id_q4_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 96,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_mul_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_mul_wg128_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_mul_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_mul_packed_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_mul_packed_2row_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_wg128_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_row2_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_row4_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_grouped_row4_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route4_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_swiglu_packed_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_id_q4_k_mul_q8_1_f32",
        "source": "mul_mat_id_q4_k_mul_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 5,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q4_k_f32",
        "source": "mul_mat_vec_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q4_k_wg128_f32",
        "source": "mul_mat_vec_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q4_k_wg64_f32",
        "source": "mul_mat_vec_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q4_k_packed_wg64_f32",
        "source": "mul_mat_vec_q4_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_quantize_q8_1_f32",
        "source": "quantize_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 56,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "pyre_quantize_q8_1_x4_f32",
        "source": "quantize_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 2,
        "constants_size": 56,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q4_k_q8_1_f32",
        "source": "mul_mat_vec_q4_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_wg64_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_cols4_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_cols8_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_cols16_wg128_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_q8_1_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_wg64_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_cols4_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_cols8_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_cols16_wg128_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_q8_1_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_q8_1_cols8_wg128_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q8_0_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": "FPIH",
        "binding_count": 3,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "pyre_mul_mat_vec_q8_0_add_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": "FPIH",
        "binding_count": 4,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
]


def write_catalog(output, entries, message):
    with output.open("w", encoding="utf-8") as f:
        f.write("// Generated by generate_pyre_kernels.py. Do not edit.\n")
        if message:
            f.write(f"// {message}\n")
        f.write('#include "kernels/pyre_kernel_catalog.h"\n\n')
        for entry in entries:
            data = entry["data"]
            f.write(f"alignas(16) static const unsigned char {entry['symbol']}[] = {{\n")
            for i in range(0, len(data), 12):
                chunk = data[i:i + 12]
                f.write("    ")
                f.write(", ".join(f"0x{value:02x}" for value in chunk))
                f.write(",\n")
            f.write("};\n\n")

        f.write("static const ggml_pyre_kernel_entry k_pyre_kernel_catalog[] = {\n")
        for entry in entries:
            wx, wy, wz = entry["workgroup_size"]
            f.write("    {\n")
            f.write(f'        "{entry["name"]}",\n')
            f.write(f"        {entry['symbol']},\n")
            f.write(f"        sizeof({entry['symbol']}),\n")
            f.write(f'        "{entry["format"]}",\n')
            f.write(f"        {entry['binding_count']},\n")
            f.write(f"        {entry['constants_size']},\n")
            f.write(f"        {{ {wx}, {wy}, {wz} }},\n")
            f.write("    },\n")
        f.write("};\n\n")
        f.write("const ggml_pyre_kernel_entry * ggml_pyre_kernel_catalog_entries(size_t * count) {\n")
        f.write("    if (count) {\n")
        f.write("        *count = sizeof(k_pyre_kernel_catalog) / sizeof(k_pyre_kernel_catalog[0]);\n")
        f.write("    }\n")
        f.write("    return k_pyre_kernel_catalog;\n")
        f.write("}\n")


def compile_kernel(clang, rocm_path, arch, source, output):
    cmd = [
        clang,
        "-x", "hip",
        "--offload-device-only",
        f"--offload-arch={arch}",
        "-O3",
        "-c", str(source),
        "-o", str(output),
    ]
    if rocm_path:
        cmd.insert(5, f"--rocm-path={rocm_path}")
    else:
        cmd.insert(5, "-nogpulib")
        cmd.insert(5, "-nogpuinc")
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang", default="")
    parser.add_argument("--rocm-path", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source_dir)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.clang:
        write_catalog(output, [], "Pyre pure-HIP kernel catalog disabled: no clang++ found.")
        return 0
    if not args.rocm_path:
        write_catalog(output, [], "Pyre pure-HIP kernel catalog disabled: no ROCm path configured.")
        return 0
    if not args.arch:
        write_catalog(output, [], "Pyre pure-HIP kernel catalog disabled: no AMDGPU architecture configured.")
        return 0

    entries = []
    with tempfile.TemporaryDirectory(prefix="ggml-pyre-kernels-") as tmp:
        tmp_dir = pathlib.Path(tmp)
        for index, kernel in enumerate(KERNELS):
            hsaco = tmp_dir / f"{kernel['name']}.hsaco"
            result = compile_kernel(args.clang, args.rocm_path, args.arch, source_dir / kernel["source"], hsaco)
            if result.returncode != 0:
                sys.stderr.write(
                    f"warning: failed to compile {kernel['name']} for {args.arch}; "
                    "emitting an empty Pyre kernel catalog\n"
                )
                sys.stderr.write(result.stderr)
                write_catalog(output, [], f"Pyre pure-HIP kernel catalog disabled: {kernel['name']} compile failed.")
                return 0

            entry = dict(kernel)
            entry["symbol"] = f"k_pyre_kernel_{index}"
            entry["data"] = hsaco.read_bytes()
            entries.append(entry)

    write_catalog(output, entries, f"Generated for AMDGPU architecture {args.arch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
