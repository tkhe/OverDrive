#pragma once

#include <torch/types.h>

namespace overdrive
{

#ifdef WITH_CUDA
    at::Tensor ms_deform_attn_forward_cuda(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int im2col_step
    );

    std::vector<at::Tensor> ms_deform_attn_backward_cuda(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int im2col_step
    );
#endif

    inline at::Tensor ms_deform_attn_forward(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int im2col_step
    )
    {
        if (value.is_cuda())
        {
#ifdef WITH_CUDA
            return ms_deform_attn_forward_cuda(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
            AT_ERROR("OverDrive is not compiled with GPU support");
#endif
        }
        AT_ERROR("MultiScale Deformable Attention is not implemented on the CPU");
    }

    inline std::vector<at::Tensor> ms_deform_attn_backward(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int im2col_step
    )
    {
        if (value.is_cuda())
        {
#ifdef WITH_CUDA
            return ms_deform_attn_backward_cuda(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
            AT_ERROR("OverDrive is not compiled with GPU support");
#endif
        }
        AT_ERROR("MultiScale Deformable Attention is not implemented on the CPU");
    }
} // namespace overdrive
