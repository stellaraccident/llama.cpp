#include <hip/hip_runtime.h>
#include <stdint.h>

struct pyre_argsort_f32_constants {
    long long ncols;
    long long nrows;
    int order;
    int ncols_pad;
};

static __device__ void pyre_swap_i32(int & a, int & b) {
    const int tmp = a;
    a = b;
    b = tmp;
}

extern "C" __global__ void pyre_argsort_f32_i32(
        const float * src, int * dst,
        pyre_argsort_f32_constants c) {
    __shared__ int indices[256];

    const int col = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    if (row >= c.nrows) {
        return;
    }

    const float * src_row = src + row * c.ncols;
    if (col < c.ncols_pad) {
        indices[col] = col;
    }
    __syncthreads();

    for (int k = 2; k <= c.ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (col < c.ncols_pad && ixj > col && ixj < c.ncols_pad) {
                const bool left_invalid = indices[col] >= c.ncols;
                const bool right_valid = indices[ixj] < c.ncols;
                const bool compare = c.order == 0 ?
                    src_row[indices[col]] > src_row[indices[ixj]] :
                    src_row[indices[col]] < src_row[indices[ixj]];
                const bool reverse_compare = c.order == 0 ?
                    src_row[indices[col]] < src_row[indices[ixj]] :
                    src_row[indices[col]] > src_row[indices[ixj]];

                if ((col & k) == 0) {
                    if (left_invalid || (right_valid && compare)) {
                        pyre_swap_i32(indices[col], indices[ixj]);
                    }
                } else if (indices[ixj] >= c.ncols || (!left_invalid && reverse_compare)) {
                    pyre_swap_i32(indices[col], indices[ixj]);
                }
            }
            __syncthreads();
        }
    }

    if (col < c.ncols) {
        dst[row * c.ncols + col] = indices[col];
    }
}
