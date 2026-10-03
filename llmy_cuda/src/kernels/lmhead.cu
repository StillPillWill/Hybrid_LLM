#include "kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace llmy {

__global__ void lm_head_kernel(const float* hidden, const float* embed_w,
                               float* logits, int V, int D) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    const float* row = embed_w + (size_t)v * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc += row[d] * hidden[d];
    logits[v] = acc;
}

// ---------------------------------------------------------------------------
// lm_head
//
// logits[V] = embed_w[V, D] @ hidden[D]
//
// embed_w is [V, D] row-major, so this is a matrix-vector product where
// the matrix is [V, D] and we want [V] = M @ v.
//
// cuBLAS stores matrices column-major. embed_w in row-major [V, D] is
// equivalent to a column-major [D, V] matrix. So we compute:
//     logits[V x 1] = embed_w_colmajor[D x V]^T @ hidden[D]
// using cublasSgemv with CUBLAS_OP_T.
// ---------------------------------------------------------------------------

void lm_head(const float* hidden,
             const float* embed_w,
             float* logits,
             int V, int D,
             cublasHandle_t blas,
             cudaStream_t s) {
    (void)blas;
    int threads = 256;
    int blocks = (V + threads - 1) / threads;
    lm_head_kernel<<<blocks, threads, 0, s>>>(hidden, embed_w, logits, V, D);
}

}  // namespace llmy
