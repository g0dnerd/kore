// y = clamp(x, 0, max_val)^2  →  dy/dx = 2*clamp(x,0,max_val) for 0<x<max_val,
// else 0 (the clamp is flat outside [0,max_val], so the chain rule kills the
// gradient there). Inside the active region clamp(x)=x, so the factor is 2*x.
__kernel void squared_clipped_relu_backward(__global const float *input,
                                            __global const float *grad_output,
                                            __global float *grad_input,
                                            const float max_val, const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    float x = input[gid];
    float d = (x > 0.0f && x < max_val) ? (2.0f * x) : 0.0f;
    grad_input[gid] += grad_output[gid] * d;
  }
}
