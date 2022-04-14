#include <cstdio>

#include "stencil_util.h"

const double ALPHA_ZZZ = 0.9415;
const double ALPHA_NZZ = 0.01531;
const double ALPHA_PZZ = 0.02345;
const double ALPHA_ZNZ = -0.01334;
const double ALPHA_ZPZ = -0.03512;
const double ALPHA_ZZN = 0.02333;
const double ALPHA_ZZP = 0.02111;
const double ALPHA_NNZ = -0.03154;
const double ALPHA_PNZ = -0.01234;
const double ALPHA_NPZ = 0.01111;
const double ALPHA_PPZ = 0.02222;
const double ALPHA_NZN = 0.01212;
const double ALPHA_PZN = 0.01313;
const double ALPHA_NZP = -0.01242;
const double ALPHA_PZP = -0.03751;
const double ALPHA_ZNN = -0.03548;
const double ALPHA_ZPN = -0.04214;
const double ALPHA_ZNP = 0.01795;
const double ALPHA_ZPP = 0.01279;
const double ALPHA_NNN = 0.01537;
const double ALPHA_PNN = -0.01357;
const double ALPHA_NPN = -0.01734;
const double ALPHA_PPN = 0.01975;
const double ALPHA_NNP = 0.02568;
const double ALPHA_PNP = 0.02734;
const double ALPHA_NPP = -0.01242;
const double ALPHA_PPP = -0.02018;

extern "C" {
extern void seven_point_256(double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double *,
                            double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

extern void seven_point_512(double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double *,
                            double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

extern void seven_point_384(double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double *,
                            double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
}

template <unsigned PAD>
__attribute__((noinline)) void seven_point_ref(double *input, double *output, int64_t offset, int64_t size_a,
                                               int64_t size_b, int64_t size_c, int64_t stride_a, int64_t stride_b,
                                               int64_t stride_c) {
#define INDEX(i, j, k) (offset + (i)*stride_a + (j)*stride_b + (k)*stride_c)
  for (int i = PAD; i < size_a - PAD; ++i) {
    for (int j = PAD; j < size_b - PAD; ++j) {
      for (int k = PAD; k < size_c - PAD; ++k) {
        output[INDEX(i, j, k)] =
            ALPHA_ZZZ * input[INDEX(i + 0, j + 0, k + 0)] + ALPHA_NZZ * input[INDEX(i - 1, j + 0, k + 0)] +
            ALPHA_PZZ * input[INDEX(i + 1, j + 0, k + 0)] + ALPHA_ZNZ * input[INDEX(i + 0, j - 1, k + 0)] +
            ALPHA_ZPZ * input[INDEX(i + 0, j + 1, k + 0)] + ALPHA_ZZN * input[INDEX(i + 0, j + 0, k - 1)] +
            ALPHA_ZZP * input[INDEX(i + 0, j + 0, k + 1)];
      }
    }
  }
#undef INDEX
}

#ifdef __NVCC__

template <unsigned PAD>
__global__ void seven_point_gpu_ref(double *input, double *output, int64_t offset, int64_t size_a, int64_t size_b,
                                    int64_t size_c, int64_t stride_a, int64_t stride_b, int64_t stride_c) {
#define INDEX(i, j, k) (offset + ((i) + PAD) * stride_a + ((j) + PAD) * stride_b + ((k) + PAD) * stride_c)
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  int k = ti % (size_c - 2 * PAD);
  int j = (ti / (size_c - 2 * PAD)) % (size_b - 2 * PAD);
  int i = (ti / (size_c - 2 * PAD)) / (size_b - 2 * PAD);

  output[INDEX(i, j, k)] =
      ALPHA_ZZZ * input[INDEX(i + 0, j + 0, k + 0)] + ALPHA_NZZ * input[INDEX(i - 1, j + 0, k + 0)] +
      ALPHA_PZZ * input[INDEX(i + 1, j + 0, k + 0)] + ALPHA_ZNZ * input[INDEX(i + 0, j - 1, k + 0)] +
      ALPHA_ZPZ * input[INDEX(i + 0, j + 1, k + 0)] + ALPHA_ZZN * input[INDEX(i + 0, j + 0, k - 1)] +
      ALPHA_ZZP * input[INDEX(i + 0, j + 0, k + 1)];
}

#endif

int main(int argc, char **argv) {
  constexpr int64_t PAD = 1;
  constexpr int64_t UB = 384;
  Square<double, 3> input(UB, PAD), output(UB, PAD), output_ref(UB, PAD);
  input.random();

  output.clear();
  output_ref.clear();

  timer.start("seven_point_384_ref");
  seven_point_ref<PAD>(input.data.get(), output_ref.data.get(), 0, output_ref.sizes[0], output_ref.sizes[1],
                       output_ref.sizes[2], output_ref.strides[0], output_ref.strides[1], output_ref.strides[2]);
  timer.stop("seven_point_384_ref");
#ifdef __NVCC__
  input.to_gpu();
  output_ref.to_gpu();
  timer.start("seven_point_384_gpu_ref");
  int block_size = 512;
  int grid_size = (UB * UB * UB + block_size - 1) / block_size;
  seven_point_gpu_ref<PAD><<<grid_size, block_size>>>(input.data_gpu, output_ref.data_gpu, 0, output_ref.sizes[0],
                                                      output_ref.sizes[1], output_ref.sizes[2], output_ref.strides[0],
                                                      output_ref.strides[1], output_ref.strides[2]);
  output_ref.sync();
  timer.stop("seven_point_384_gpu_ref");
  output_ref.to_cpu();

  input.to_gpu();
  output.to_gpu();
  timer.start("seven_point_384");
  seven_point_384(input.data_gpu, input.data_gpu, 0, input.sizes[0], input.sizes[1], input.sizes[2], input.strides[0],
                  input.strides[1], input.strides[2], output.data_gpu, output.data_gpu, 0, output.sizes[0],
                  output.sizes[1], output.sizes[2], output.strides[0], output.strides[1], output.strides[2]);
  timer.stop("seven_point_384");
  output.to_cpu();

#else
  timer.start("seven_point_384");
  seven_point_384(input.data.get(), input.data.get(), 0, input.sizes[0], input.sizes[1], input.sizes[2],
                  input.strides[0], input.strides[1], input.strides[2], output.data.get(), output.data.get(), 0,
                  output.sizes[0], output.sizes[1], output.sizes[2], output.strides[0], output.strides[1],
                  output.strides[2]);
  timer.stop("seven_point_384");
#endif
  assert(output_ref == output);
  std::cout << "7point 384 PASS" << std::endl;
  timer.show_all();
  return 0;
}
