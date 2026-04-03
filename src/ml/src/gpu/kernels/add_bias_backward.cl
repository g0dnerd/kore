// grad_bias[j] += sum_i grad_output[i * cols + j]
// Column-wise reduction over rows.
__kernel void add_bias_backward(__global const float *grad_output,
                                __global float *grad_bias, const uint rows,
                                const uint cols) {
  const uint j = get_global_id(0);
  if (j >= cols)
    return;

  float sum = 0.0f;
  for (uint i = 0; i < rows; i++) {
    sum += grad_output[i * cols + j];
  }
  grad_bias[j] += sum;
}
