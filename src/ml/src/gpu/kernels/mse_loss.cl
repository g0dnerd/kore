__kernel void mse_loss(__global const float *pred, __global const float *target,
                       __global float *element_loss, __global float *grad,
                       const unsigned int n, const float scale,
                       const float inv_n) {
  int i = get_global_id(0);
  if (i >= n)
    return;

  float p = pred[i];
  float t = target[i];
  float f_p, f_prime;

  if (scale > 0.0f) {
    float sp = 1.0f / (1.0f + exp(-p * scale));
    f_p = sp;
    f_prime = scale * sp * (1.0f - sp);
  } else {
    f_p = p;
    f_prime = 1.0f;
  }

  float diff = f_p - t;
  element_loss[i] = diff * diff;
  grad[i] = 2.0f * inv_n * diff * f_prime;
}
