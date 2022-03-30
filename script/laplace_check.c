#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void main_kernel(double *, double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double *,
                        double *, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

#define IDX(i, j, k) (((i)*64 + (j)) * 64 + (k))
double lap_factor = -4.0;
void laplace(double *input, double *output) {
  for (int i = 1; i < 63; ++i) {
    for (int j = 1; j < 63; ++j) {
      for (int k = 1; k < 63; ++k) {
        output[IDX(i, j, k)] = lap_factor * input[IDX(i, j, k)] + input[IDX(i - 1, j, k)] + input[IDX(i + 1, j, k)] +
                               input[IDX(i, j + 1, k)] + input[IDX(i, j - 1, k)];
      }
    }
  }
}

void set(double *input, double v) {
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      for (int k = 0; k < 64; ++k) {
        input[IDX(i, j, k)] = v;
      }
    }
  }
}

void show(double *input) {
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      for (int k = 0; k < 64; ++k) {
        printf("[%2d, %2d, %2d] = %lf\n", i, j, k, input[(i * 64 + j) * 64 + k]);
      }
    }
  }
}

void check(double *input, double *output) {
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      for (int k = 0; k < 64; ++k) {
        if (fabs(input[IDX(i, j, k)] - output[IDX(i, j, k)]) >= 1e-6) {
          printf("ERR at (%d, %d, %d)\n", i, j, k);
          return;
        }
      }
    }
  }
  printf("PASS\n");
}

int main() {
  double *in = (double *)malloc(sizeof(double) * (64 * 64 * 64));
  double *out = (double *)malloc(sizeof(double) * (64 * 64 * 64));
  double *out_ref = (double *)malloc(sizeof(double) * (64 * 64 * 64));
  set(in, 3.1415);
  set(out, 0);
  set(out_ref, 0);
  main_kernel(in, in, 0, 0, 0, 0, 0, 0, 0, out, out, 0, 0, 0, 0, 0, 0, 0);
  check(out, out_ref);
  return 0;
}
