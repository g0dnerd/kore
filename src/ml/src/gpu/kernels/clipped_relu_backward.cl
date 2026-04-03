// grad_input[i] += grad_output[i] * (0 < input[i] < max_val ? 1 : 0)
__kernel void clipped_relu_backward(__global const float *input,
                                    __global const float *grad_output,
                                    __global float *grad_input,
                                    const float max_val, const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    float x = input[gid];
    float mask = (x > 0.0f && x < max_val) ? 1.0f : 0.0f;
    grad_input[gid] += grad_output[gid] * mask;
  }
}
