__kernel void clipped_relu(__global const float *input, __global float *output,
                           const float max_val, const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    output[gid] = fmin(fmax(input[gid], 0.0f), max_val);
  }
}
