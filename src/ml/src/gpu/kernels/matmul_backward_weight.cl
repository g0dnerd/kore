#define TS 16

// grad_weight[K,N] += input^T[K,M] * grad_output[M,N]
// Tiled matmul with first operand transposed.
__kernel void matmul_backward_weight(__global const float *input,
                                     __global const float *grad_output,
                                     __global float *grad_weight, const uint M,
                                     const uint N, const uint K) {
  __local float tile_it[TS][TS];
  __local float tile_go[TS][TS];

  const uint row = get_local_id(0);
  const uint col = get_local_id(1);
  const uint global_row = get_group_id(0) * TS + row; // k in [0, K)
  const uint global_col = get_group_id(1) * TS + col; // n in [0, N)

  float acc = 0.0f;
  const uint num_tiles = (M + TS - 1) / TS;

  for (uint t = 0; t < num_tiles; t++) {
    // input^T[global_row, t*TS+col] = input[t*TS+col, global_row]
    const uint i_row = t * TS + col;
    if (global_row < K && i_row < M)
      tile_it[row][col] = input[i_row * K + global_row];
    else
      tile_it[row][col] = 0.0f;

    // grad_output[t*TS+row, global_col]
    const uint go_row = t * TS + row;
    if (go_row < M && global_col < N)
      tile_go[row][col] = grad_output[go_row * N + global_col];
    else
      tile_go[row][col] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint m = 0; m < TS; m++) {
      acc += tile_it[row][m] * tile_go[m][col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < K && global_col < N)
    grad_weight[global_row * N + global_col] += acc;
}
