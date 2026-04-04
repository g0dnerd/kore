__kernel void concat_backward(__global const float *grad_output,
                              __global float *grad_a, __global float *grad_b,
                              const uint rows, const uint cols_a,
                              const uint cols_b) {
  const uint row = get_global_id(0);
  const uint col = get_global_id(1);
  const uint cols_out = cols_a + cols_b;
  if (row >= rows || col >= cols_out)
    return;

  if (col < cols_a) {
    grad_a[row * cols_a + col] += grad_output[row * cols_out + col];
  } else {
    grad_b[row * cols_b + (col - cols_a)] += grad_output[row * cols_out + col];
  }
}
