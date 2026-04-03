__kernel void concat(__global const float *a, __global const float *b,
                     __global float *output, const uint rows, const uint cols_a,
                     const uint cols_b) {
  const uint row = get_global_id(0);
  const uint col = get_global_id(1);
  const uint cols_out = cols_a + cols_b;
  if (row >= rows || col >= cols_out)
    return;

  if (col < cols_a) {
    output[row * cols_out + col] = a[row * cols_a + col];
  } else {
    output[row * cols_out + col] = b[row * cols_b + (col - cols_a)];
  }
}
