#define TS 16

__kernel void matmul(__global const float *A, __global const float *B,
                     __global float *C, const uint M, const uint N,
                     const uint K) {
  __local float tile_a[TS][TS];
  __local float tile_b[TS][TS];

  const uint row = get_local_id(0);
  const uint col = get_local_id(1);
  const uint global_row = get_group_id(0) * TS + row;
  const uint global_col = get_group_id(1) * TS + col;

  float acc = 0.0f;
  const uint num_tiles = (K + TS - 1) / TS;

  for (uint t = 0; t < num_tiles; t++) {
    const uint a_col = t * TS + col;
    if (global_row < M && a_col < K)
      tile_a[row][col] = A[global_row * K + a_col];
    else
      tile_a[row][col] = 0.0f;

    const uint b_row = t * TS + row;
    if (b_row < K && global_col < N)
      tile_b[row][col] = B[b_row * N + global_col];
    else
      tile_b[row][col] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < TS; k++) {
      acc += tile_a[row][k] * tile_b[k][col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < M && global_col < N)
    C[global_row * N + global_col] = acc;
}
