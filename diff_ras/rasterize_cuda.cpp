#include <torch/torch.h>
#include <vector>
#include <iostream>

#include "rasterize_cuda_kernel.h"

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> forward_rasterize(
        at::Tensor vertices,
        at::Tensor rasterized,
        at::Tensor contribution_map,
        int width,
        int height,
        float inv_smoothness,
        int mode) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(rasterized);
    CHECK_INPUT(contribution_map);

    return forward_rasterize_cuda(vertices, rasterized, contribution_map, width, height, inv_smoothness, mode);
}

at::Tensor backward_rasterize(
        at::Tensor vertices,
        at::Tensor rasterized,
        at::Tensor contribution_map,
        at::Tensor grad_output,
        at::Tensor grad_vertices,
        int width,
        int height,
        float inv_smoothness,
        int mode) {        
    CHECK_INPUT(vertices);
    CHECK_INPUT(rasterized);
    CHECK_INPUT(contribution_map);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_vertices);

    return backward_rasterize_cuda(vertices, rasterized, contribution_map, grad_output, grad_vertices, width, height, inv_smoothness, mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_rasterize", &forward_rasterize, "forward rasterize (CUDA)");
    m.def("backward_rasterize", &backward_rasterize, "backward rasterize (CUDA)");
}
