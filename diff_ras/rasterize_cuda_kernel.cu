#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "rasterize_cuda_kernel.h"

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#define GET_DIRECT_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3)         \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3)])

#define ADD_ATOMIC_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3, v)        \
  atomicAdd( data + (x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3), v )
#define ADD_ATOMIC_5d(data, x0, x1, x2, x3, x4, sd0, sd1, sd2, sd3, sd4, v) \
  atomicAdd( data + (x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3) + (x4)*(sd4), v )
#define SET_DIRECT_4d(data, x0, x1, x2, x3, sd0, sd1, sd2, sd3, v)        \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) + (x3) * (sd3)]) = v

#define GET_DIRECT_3d(data, x0, x1, x2, sd0, sd1, sd2) \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2)])

#define SET_DIRECT_3d(data, x0, x1, x2, sd0, sd1, sd2, v)        \
  ((data)[(x0) * (sd0) + (x1) * (sd1) + (x2) * (sd2) ]) = v

#define GET_DIRECT_5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)])

#define SET_DIRECT_5d(data, x0, x1, x2, x3, x4, stride0, stride1, stride2, stride3, stride4, value) \
  ((data)[(x0)*(stride0)+(x1)*(stride1)+(x2)*(stride2)+(x3)*(stride3)+(x4)*(stride4)] = (value))

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


const int MODE_BOUNDARY = 0;
const int MODE_MASK = 1;
const int MODE_HARD_MASK = 2;

template <typename scalar_t>
__global__ void inside_outside_cuda_kernel(
        const scalar_t* __restrict__ vertices,
        int batch_size,
        int number_vertices,
        scalar_t* rasterized,
        int height,
        int width) {
    // 1-D array of 1-D blocks.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * width * height) {
        return;
    }
    
    const int w = width;
    const int h = height;
    const int nv = number_vertices;

    // batch index.
    const int bi = i / (w * h);

    // pixel number (linear index)
    const int pn = i % (w * h);
    const int yp = pn / w;
    const int xp = pn % w;

    // cast a ray: William Randolph Franklin.
    int j = 0;
    scalar_t c = 0;
    for (int vn = 0, j = nv - 1; vn < nv; j = vn++) {
      scalar_t from_x;
      scalar_t from_y;
      scalar_t to_x;
      scalar_t to_y;

      from_x = vertices[bi * (nv * 2) + vn * 2];
      from_y = vertices[bi * (nv * 2) + vn * 2 + 1];
      to_x = vertices[bi * (nv * 2) + j * 2];
      to_y = vertices[bi * (nv * 2) + j * 2 + 1];
      
      if (((from_y > yp) != (to_y > yp)) && (xp < (to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x)) {
        c = !c;
      }
    }

    rasterized[i] = c == 0 ? -1.0 : 1.0;
}

template <typename scalar_t>
__global__ void forward_rasterize_cuda_kernel(
        const scalar_t* __restrict__ vertices,
        int batch_size,
        int number_vertices,
        scalar_t* rasterized,
        int* contribution_map,
        int height,
        int width,
        float inv_smoothness,
        int mode) {
    // 1-D array of 1-D blocks.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * width * height) {
        return;
    }
    
    const int w = width;
    const int h = height;
    const int nv = number_vertices;

    // batch index.
    const int bi = i / (w * h);

    // pixel number (linear index)
    const int pn = i % (w * h);
    const int yp = pn / w;
    const int xp = pn % w;

    // go through each vertex.
    // at some point, we'll need to record
    // which segment contributed the most
    // for backwards pass.
    scalar_t max_contribution = -2147483647;
    int max_vertex_number = -1;
    for (int vn = 0; vn < nv; vn++) {
        int from_index;
        int to_index;
        scalar_t from_x;
        scalar_t from_y;
        scalar_t to_x;
        scalar_t to_y;
        scalar_t x2_sub_x1;
        scalar_t y2_sub_y1;
        scalar_t square_segment_length;
        scalar_t x_sub_x1;
        scalar_t y_sub_y1;
        scalar_t x_sub_x2;
        scalar_t y_sub_y2;
        scalar_t dot;
        scalar_t x_proj;
        scalar_t y_proj;
        scalar_t contribution;

        // grid_x, grid_y = xp, yp.
        from_index = vn;
        to_index = (vn + 1) % number_vertices;

        from_x = vertices[bi * (nv * 2) + from_index * 2];
        from_y = vertices[bi * (nv * 2) + from_index * 2 + 1];

        to_x = vertices[bi * (nv * 2) + to_index * 2];
        to_y = vertices[bi * (nv * 2) + to_index * 2 + 1];
        
        x2_sub_x1 = to_x - from_x;
        y2_sub_y1 = to_y - from_y;
            
        square_segment_length = x2_sub_x1 * x2_sub_x1 + y2_sub_y1 * y2_sub_y1 + 0.00001;

        x_sub_x1 = xp - from_x;
        y_sub_y1 = yp - from_y;
        x_sub_x2 = xp - to_x;
        y_sub_y2 = yp - to_y;

        dot = ((x_sub_x1 * x2_sub_x1) + (y_sub_y1 * y2_sub_y1)) / square_segment_length;
        x_proj = xp - (from_x + dot * x2_sub_x1);
        y_proj = yp - (from_y + dot * y2_sub_y1);

        // Does it matter here to compute the squared distance or true Euclidean distance?
        if (dot < 0) {
          contribution = pow(x_sub_x1, 2) + pow(y_sub_y1, 2);
        }
        else if (dot > 1) {
          contribution = pow(x_sub_x2, 2) + pow(y_sub_y2, 2);
        }
        else {
          contribution = pow(x_proj, 2) + pow(y_proj, 2);
        }

        // we need contribution to be a decreasing function.
        // if (mode == MODE_MASK) {
        //     // sign * -dist
        //     contribution = 1.0 / (1.0 + exp(-rasterized[i] * contribution / inv_smoothness));
        // }
        // else if (mode == MODE_HARD_MASK) {
        //     // map the inside outside map to 0 or 1.0.
        //     // technically, we don't need this preceeding loop.
        //     contribution = rasterized[i] < 0 ? 0.0 : 1.0;
        // }
        // else {
        //     contribution = exp(-contribution / inv_smoothness);
        // }
        
        contribution = -contribution;
        
        if (contribution > max_contribution) {
          max_contribution = contribution;
          max_vertex_number = vn;
        }
    }

    if (mode == MODE_MASK) {
        // sign * -dist
        max_contribution = 1.0 / (1.0 + exp(rasterized[i] * max_contribution / inv_smoothness));
    }
    else if (mode == MODE_HARD_MASK) {
        // map the inside outside map to 0 or 1.0.
        // technically, we don't need this preceeding loop.
        max_contribution = rasterized[i] < 0 ? 0.0 : 1.0;
    }
    else {
        max_contribution = exp(max_contribution / inv_smoothness);
    }

    rasterized[i] = max_contribution;
    contribution_map[i] = max_vertex_number;
}

template <typename scalar_t>
__global__ void backward_rasterize_cuda_kernel(
        const scalar_t* __restrict__ vertices,
        const scalar_t* __restrict__ rasterized,
        const int* __restrict__ contribution_map,
        const scalar_t* __restrict__ grad_output,
        scalar_t* grad_vertices,
        int batch_size,
        int number_vertices,
        int width,
        int height,
        float inv_smoothness) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * width * height) {
        return;
    }

    const int w = width;
    const int h = height;
    const int nv = number_vertices;

    // batch index.
    const int bi = i / (w * h);

    // pixel number (linear index)
    const int pn = i % (w * h);
    const int yp = pn / w;
    const int xp = pn % w;

    // produce dR/dv.
    // since we use max over all vertices, we only need
    // to apply it to single vertex.
    int vn;
    int from_index;
    int to_index;
    scalar_t from_x;
    scalar_t from_y;
    scalar_t to_x;
    scalar_t to_y;
    scalar_t x2_sub_x1;
    scalar_t y2_sub_y1;
    scalar_t square_segment_length;
    scalar_t x_sub_x1;
    scalar_t y_sub_y1;
    scalar_t x_sub_x2;
    scalar_t y_sub_y2;
    scalar_t dot;
    scalar_t x_proj;
    scalar_t y_proj;
    scalar_t grad_x1 = 0.0;
    scalar_t grad_y1 = 0.0;
    scalar_t grad_x2 = 0.0;
    scalar_t grad_y2 = 0.0;

    scalar_t in_out = rasterized[i] >= 0.5 ? 1.0 : -1.0;

    vn = contribution_map[i];
    from_index = vn;
    to_index = (vn + 1) % nv;
    
    // determine how we computed the distance to this segment.
    from_x = vertices[bi * (nv * 2) + from_index * 2];
    from_y = vertices[bi * (nv * 2) + from_index * 2 + 1];

    to_x = vertices[bi * (nv * 2) + to_index * 2];
    to_y = vertices[bi * (nv * 2) + to_index * 2 + 1];
        
    x2_sub_x1 = to_x - from_x;
    y2_sub_y1 = to_y - from_y;

    // grad:
    // dX1 = 2 * x2_sub_x1 * -1
    // dX2 = 2 * x2_sub_x1
    // dY1 = 2 * y2_sub_y1 * -1
    // dY2 = 2 * y2_sub_y1
    // possible this could NaN?
    square_segment_length = x2_sub_x1 * x2_sub_x1 + y2_sub_y1 * y2_sub_y1 + 0.00001;

    x_sub_x1 = xp - from_x;
    y_sub_y1 = yp - from_y;
    x_sub_x2 = xp - to_x;
    y_sub_y2 = yp - to_y;

    // grad numer:
    // dX1 = -1 * x2_sub_x1 + -1 * x_sub_x1
    // dX2 = x_sub_x1
    scalar_t dot_num = ((x_sub_x1 * x2_sub_x1) + (y_sub_y1 * y2_sub_y1));
    dot = dot_num / square_segment_length;
    x_proj = xp - (from_x + dot * x2_sub_x1);
    y_proj = yp - (from_y + dot * y2_sub_y1);

    // negative sign?
    if (dot < 0) {
        // contribution = exp(-((xp - from_x) ** 2 + (yp - from_y) ** 2 / inv_smoothness)
        // grad_x1 = (rasterized[i] * 2 * x_sub_x1) / inv_smoothness;
        // grad_y1 = (rasterized[i] * 2 * y_sub_y1) / inv_smoothness;
        // grad_x1 = in_out * rasterized[i] * (1.0 - rasterized[i]) * 2 * x_sub_x1 / inv_smoothness;
        // grad_y1 = in_out * rasterized[i] * (1.0 - rasterized[i]) * 2 * y_sub_y1 / inv_smoothness;
        grad_x1 = in_out * rasterized[i] * (1.0 - rasterized[i]) * -2 * x_sub_x1 / inv_smoothness;
        grad_y1 = in_out * rasterized[i] * (1.0 - rasterized[i]) * -2 * y_sub_y1 / inv_smoothness;
    }
    else if (dot > 1) {
      // contribution = exp(-((xp - to_x) ** 2 + (yp - to_y) ** 2) / inv_smoothness)
        // grad_x2 = (rasterized[i] * 2 * x_sub_x2) / inv_smoothness;
        // grad_y2 = (rasterized[i] * 2 * y_sub_y2) / inv_smoothness;
        grad_x2 = in_out * rasterized[i] * (1.0 - rasterized[i]) * -2 * x_sub_x2 / inv_smoothness;
        grad_y2 = in_out * rasterized[i] * (1.0 - rasterized[i]) * -2 * y_sub_y2 / inv_smoothness;
    }
    else {
      // contribution = exp(-(xp - from_x) ** 2 / inv_smoothness)
      scalar_t ss_x1 = -2.0 * x2_sub_x1;
      scalar_t ss_x2 = 2.0 * x2_sub_x1;
      scalar_t ss_y1 = -2.0 * y2_sub_y1;
      scalar_t ss_y2 = 2.0 * y2_sub_y1;

      scalar_t dot_x1 = (square_segment_length * (-x2_sub_x1 - x_sub_x1) - dot_num * ss_x1) / pow(square_segment_length, 2);
      scalar_t dot_x2 = (square_segment_length * x_sub_x1 - dot_num * ss_x2) / pow(square_segment_length, 2);
      scalar_t dot_y1 = (square_segment_length * (-y2_sub_y1 - y_sub_y1) - dot_num * ss_y1) / pow(square_segment_length, 2);
      scalar_t dot_y2 = (square_segment_length * y_sub_y1 - dot_num * ss_y2) / pow(square_segment_length, 2);

      // d/dx()
      scalar_t x_proj_x1 = -1 - dot_x1 * x2_sub_x1 + dot;
      scalar_t x_proj_x2 = -(dot_x2 * x2_sub_x1 + dot);

      scalar_t y_proj_y1 = -1 - dot_y1 * y2_sub_y1 + dot;
      scalar_t y_proj_y2 = -(dot_y2 * y2_sub_y1 + dot);

      // we also need mixed.
      scalar_t y_proj_x1 = -dot_x1 * y2_sub_y1;
      scalar_t y_proj_x2 = -dot_x2 * y2_sub_y1;
      scalar_t x_proj_y1 = -dot_y1 * x2_sub_x1;
      scalar_t x_proj_y2 = -dot_y2 * x2_sub_x1;

      // - as well?
      grad_x1 = in_out * rasterized[i] * (1.0 - rasterized[i]) * (2.0 * x_proj * x_proj_x1 + 2.0 * y_proj * y_proj_x1) / inv_smoothness;
      grad_x2 = in_out * rasterized[i] * (1.0 - rasterized[i]) * (2.0 * x_proj * x_proj_x2 + 2.0 * y_proj * y_proj_x2) / inv_smoothness;
      grad_y1 = in_out * rasterized[i] * (1.0 - rasterized[i]) * (2.0 * x_proj * x_proj_y1 + 2.0 * y_proj * y_proj_y1) / inv_smoothness;
      grad_y2 = in_out * rasterized[i] * (1.0 - rasterized[i]) * (2.0 * x_proj * x_proj_y2 + 2.0 * y_proj * y_proj_y2) / inv_smoothness;
      
      // grad_x1 = -rasterized[i] * (2.0 * x_proj * x_proj_x1 + 2.0 * y_proj * y_proj_x1) / inv_smoothness;
      // grad_x2 = -rasterized[i] * (2.0 * x_proj * x_proj_x2 + 2.0 * y_proj * y_proj_x2) / inv_smoothness;
      // grad_y1 = -rasterized[i] * (2.0 * x_proj * x_proj_y1 + 2.0 * y_proj * y_proj_y1) / inv_smoothness;
      // grad_y2 = -rasterized[i] * (2.0 * x_proj * x_proj_y2 + 2.0 * y_proj * y_proj_y2) / inv_smoothness;      
    }

    // apply the input gradients.
    grad_x1 = grad_x1 * grad_output[i];
    grad_x2 = grad_x2 * grad_output[i];
    grad_y1 = grad_y1 * grad_output[i];
    grad_y2 = grad_y2 * grad_output[i];

    // grad_vertices[bi * (height * width * nv * 2) + yp * (width * nv * 2) + xp * (nv * 2) + from_index * 2] = grad_x1;
    // grad_vertices[bi * (height * width * nv * 2) + yp * (width * nv * 2) + xp * (nv * 2) + from_index * 2 + 1] = grad_y1;
    // grad_vertices[bi * (height * width * nv * 2) + yp * (width * nv * 2) + xp * (nv * 2) + to_index * 2] = grad_x2;
    // grad_vertices[bi * (height * width * nv * 2) + yp * (width * nv * 2) + xp * (nv * 2) + to_index * 2 + 1] = grad_y2;

    // unsure if should be deferencing.
    atomicAdd(grad_vertices + bi * nv * 2 + from_index * 2, grad_x1);
    atomicAdd(grad_vertices + bi * nv * 2 + from_index * 2 + 1, grad_y1);
    atomicAdd(grad_vertices + bi * nv * 2 + to_index * 2, grad_x2);
    atomicAdd(grad_vertices + bi * nv * 2 + to_index * 2 + 1, grad_y2);
}

std::vector<at::Tensor> forward_rasterize_cuda(
        at::Tensor vertices,
        at::Tensor rasterized,
        at::Tensor contribution_map,
        int width,
        int height,
        float inv_smoothness,
        int mode) {
    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);
    const int threads = 512;

    // each block processes some 512 sized chunk of the output image.
    const dim3 blocks ((batch_size * width * height - 1) / threads + 1);

    if ((mode == MODE_MASK) || (mode == MODE_HARD_MASK)) {
      // determine whether each point is inside or outside.
      AT_DISPATCH_FLOATING_TYPES(vertices.type(), "inside_outside_cuda", ([&] {
            inside_outside_cuda_kernel<scalar_t><<<blocks, threads>>>(
              vertices.data<scalar_t>(),
              batch_size,
              number_vertices,
              rasterized.data<scalar_t>(),
              height,
              width);
          }));
    }

    if (mode != MODE_HARD_MASK) {
        AT_DISPATCH_FLOATING_TYPES(vertices.type(), "forward_rasterize_cuda", ([&] {
        forward_rasterize_cuda_kernel<scalar_t><<<blocks, threads>>>(
            vertices.data<scalar_t>(),
	    batch_size,
            number_vertices,
            rasterized.data<scalar_t>(),
            contribution_map.data<int>(),
            height,
            width,
            inv_smoothness,
            mode);
        }));
     }

    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in forward_rasterize: %s\n", cudaGetErrorString(err));

    return { rasterized, contribution_map };
}

at::Tensor backward_rasterize_cuda(
        at::Tensor vertices,
        at::Tensor rasterized,
        at::Tensor contribution_map,
        at::Tensor grad_output,
        at::Tensor grad_vertices,
        int width,
        int height,
        float inv_smoothness,
        int mode) {
    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);
    const int threads = 512;
    const dim3 blocks ((batch_size * width * height - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(vertices.type(), "backward_rasterize_cuda", ([&] {
      backward_rasterize_cuda_kernel<scalar_t><<<blocks, threads>>>(
          vertices.data<scalar_t>(),
          rasterized.data<scalar_t>(),
          contribution_map.data<int>(),
          grad_output.data<scalar_t>(),
          grad_vertices.data<scalar_t>(),
          batch_size,
          number_vertices,
          width,
          height,
          inv_smoothness);
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in backward_rasterize: %s\n", cudaGetErrorString(err));

    return grad_vertices;
}
