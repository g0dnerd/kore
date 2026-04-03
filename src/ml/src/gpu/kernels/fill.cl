__kernel void fill(__global float *buf, const float value, const uint n) {
  const uint gid = get_global_id(0);
  if (gid < n) {
    buf[gid] = value;
  }
}
