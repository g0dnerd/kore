__kernel void quantize_i8(__global const float *input, __global char *output,
                          const float scale, const uint n) {
  uint i = get_global_id(0);
  if (i >= n)
    return;
  float val = input[i] * scale;
  val = clamp(val, -128.0f, 127.0f);
  output[i] = (char)(int)round(val);
}
