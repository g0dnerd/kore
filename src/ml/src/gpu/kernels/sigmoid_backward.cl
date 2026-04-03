// grad_input[i] += grad_output[i] * s[i] * (1 - s[i])
// where s = sigmoid output from forward pass.
__kernel void sigmoid_backward(__global const float *sigmoid_output,
                               __global const float *grad_output,
                               __global float *grad_input, const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    float s = sigmoid_output[gid];
    grad_input[gid] += grad_output[gid] * s * (1.0f - s);
  }
}
