#include <cstdio>

#include "stencil_util.h"

//#include "puzzle_laplacian.h"
extern "C" {
extern void laplacian(double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t, double *, double *, int64_t,
                      int64_t, int64_t, int64_t, int64_t);
}

const double lap_factor = 4.0;
template <unsigned PAD>
void laplacian_ref(double *phi, double *lap, int64_t offset, int64_t size_a, int64_t size_b, int64_t stride_a,
                   int64_t stride_b) {
#define INDEX(i, j) (offset + (i)*stride_a + (j)*stride_b)
  for (int i = PAD; i < size_a - PAD; ++i) {
    for (int j = PAD; j < size_b - PAD; ++j) {
      lap[INDEX(i, j)] = phi[INDEX(i + 1, j)] + phi[INDEX(i - 1, j)] + phi[INDEX(i, j + 1)] + phi[INDEX(i, j - 1)] -
                         lap_factor * phi[INDEX(i, j)];
    }
  }
}

int main(int argc, char **argv) {
  constexpr int64_t PAD = 1;
  constexpr int64_t UB = 64;
  Square<double, 2> in(UB, PAD), out(UB, PAD), out_ref(UB, PAD);
  in.random();
  out.clear();
  out_ref.clear();

  assert(out == out_ref);
  laplacian_ref<PAD>(in.data.get(), out.data.get(), 0, out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]);
  assert(!(out == out_ref));
  laplacian(in.data.get(), in.data.get(), 0, in.sizes[0], in.sizes[1], in.strides[0], in.strides[1], out_ref.data.get(),
            out_ref.data.get(), 0, out_ref.sizes[0], out_ref.sizes[1], out_ref.strides[0], out_ref.strides[1]);
  assert(out == out_ref);
  std::cout << "laplacian PASS" << std::endl;
  return 0;
}
