__kernel void reduce_sum(__global const float *input, __global float *output,
                         __local float *scratch, const uint n) {
  const uint gid = get_global_id(0);
  const uint lid = get_local_id(0);
  const uint group_size = get_local_size(0);
  const uint group_id = get_group_id(0);

  scratch[lid] = (gid < n) ? input[gid] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint s = group_size / 2; s > 0; s >>= 1) {
    if (lid < s) {
      scratch[lid] += scratch[lid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    output[group_id] = scratch[0];
  }
}
