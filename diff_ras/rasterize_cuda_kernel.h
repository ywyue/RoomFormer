#ifdef __cplusplus
extern "C" {
  #endif

  // CUDA forward declarations
  std::vector<at::Tensor> forward_rasterize_cuda(at::Tensor vertices,
                                                 at::Tensor rasterized,
                                                 at::Tensor contribution_map,
                                                 int width,
                                                 int height,
                                                 float inv_smoothness,
                                                 int mode);

  at::Tensor backward_rasterize_cuda(at::Tensor vertices,
                                     at::Tensor rasterized,
                                     at::Tensor contribution_map,
                                     at::Tensor grad_output,
                                     at::Tensor grad_vertices,
                                     int width,
                                     int height,
                                     float inv_smoothness,
                                     int mode);

  #ifdef __cplusplus
}
#endif
