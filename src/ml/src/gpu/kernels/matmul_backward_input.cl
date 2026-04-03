#define TS 16

// grad_input[M,K] += grad_output[M,N] * weight^T[N,K]
// Tiled matmul with second operand transposed.
__kernel void matmul_backward_input(__global const float *grad_output,
                                    __global const float *weight,
                                    __global float *grad_input, const uint M,
                                    const uint N, const uint K) {
  __local float tile_go[TS][TS];
  __local float tile_wt[TS][TS];

  const uint row = get_local_id(0);
  const uint col = get_local_id(1);
  const uint global_row = get_group_id(0) * TS + row;
  const uint global_col = get_group_id(1) * TS + col;

  float acc = 0.0f;
  const uint num_tiles = (N + TS - 1) / TS;

  for (uint t = 0; t < num_tiles; t++) {
    const uint go_col = t * TS + col;
    if (global_row < M && go_col < N)
      tile_go[row][col] = grad_output[global_row * N + go_col];
    else
      tile_go[row][col] = 0.0f;

    // weight^T[t*TS+row, global_col] = weight[global_col, t*TS+row]
    const uint w_row = t * TS + row;
    if (w_row < N && global_col < K)
      tile_wt[row][col] = weight[global_col * N + w_row];
    else
      tile_wt[row][col] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint n = 0; n < TS; n++) {
      acc += tile_go[row][n] * tile_wt[n][col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < M && global_col < K)
    grad_input[global_row * K + global_col] += acc;
}
