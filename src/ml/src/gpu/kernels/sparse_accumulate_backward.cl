// Scatter-add gradients into active weight rows.
// grad_weights[active[b][i]][j] += grad_output[b][j]

inline void atomicAdd_f(__global float *addr, float val) {
  float current = *addr;
  float expected;
  do {
    expected = current;
    float desired = expected + val;
    uint old = atomic_cmpxchg((__global uint *)addr, as_uint(expected),
                              as_uint(desired));
    current = as_float(old);
  } while (as_uint(current) != as_uint(expected));
}

__kernel void sparse_accumulate_backward(
    __global float *grad_weights, __global const float *grad_output,
    __global const uint *active_indices, __global const uint *num_active,
    const uint batch_size, const uint max_active, const uint out_features) {
  const uint b = get_global_id(0);
  const uint j = get_global_id(1);
  if (b >= batch_size || j >= out_features)
    return;

  const uint n = num_active[b];
  const float go = grad_output[b * out_features + j];

  for (uint i = 0; i < n; i++) {
    const uint idx = active_indices[b * max_active + i];
    atomicAdd_f(&grad_weights[idx * out_features + j], go);
  }
}
