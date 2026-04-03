// out[i] = a[i] * alpha + b[i] * beta
// a, b, out may alias (e.g. in-place scaling: out = a, beta = 0).
__kernel void weighted_add(__global const float *a, __global const float *b,
                           __global float *out, const float alpha,
                           const float beta, const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    out[gid] = a[gid] * alpha + b[gid] * beta;
  }
}
