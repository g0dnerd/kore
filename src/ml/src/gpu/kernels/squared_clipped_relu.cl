// output[i] = clamp(input[i], 0, max_val)^2
__kernel void squared_clipped_relu(__global const float *input,
                                   __global float *output, const float max_val,
                                   const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    float c = fmin(fmax(input[gid], 0.0f), max_val);
    output[gid] = c * c;
  }
}
