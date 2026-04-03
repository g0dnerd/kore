__kernel void sparse_accumulate(__global const float *weights,
                                __global const float *bias,
                                __global const uint *active_indices,
                                __global const uint *num_active,
                                __global float *output, const uint batch_size,
                                const uint max_active,
                                const uint out_features) {
  const uint b = get_global_id(0);
  const uint j = get_global_id(1);
  if (b >= batch_size || j >= out_features)
    return;

  float acc = bias[j];
  const uint n = num_active[b];
  for (uint i = 0; i < n; i++) {
    const uint idx = active_indices[b * max_active + i];
    acc += weights[idx * out_features + j];
  }
  output[b * out_features + j] = acc;
}
