__kernel void sigmoid(__global const float *input, __global float *output,
                      const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
  }
}
