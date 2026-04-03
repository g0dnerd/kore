__kernel void add_bias(__global float *output, __global const float *bias,
                       const uint rows, const uint cols) {
  const uint gid = get_global_id(0);
  if (gid < rows * cols) {
    output[gid] += bias[gid % cols];
  }
}
