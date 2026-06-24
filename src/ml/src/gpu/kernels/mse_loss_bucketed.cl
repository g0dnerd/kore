// Bucketed MSE loss: predictions are [batch, num_buckets]; for each sample only
// the column selected by bucket[b] contributes to the loss and receives gradient.
// Dispatched over all batch*num_buckets elements so every grad/element_loss entry
// is written (non-selected columns are explicitly zeroed). inv_n == 1/batch, so the
// reduced element_loss sum is the per-sample mean, matching the scalar-head loss.
__kernel void mse_loss_bucketed(__global const float *pred,
                                __global const float *target,
                                __global const unsigned int *bucket,
                                __global float *element_loss, __global float *grad,
                                const unsigned int total,
                                const unsigned int num_buckets, const float scale,
                                const float inv_n) {
  int i = get_global_id(0);
  if (i >= total)
    return;

  const unsigned int b = (unsigned int)i / num_buckets;
  const unsigned int j = (unsigned int)i % num_buckets;

  if (j != bucket[b]) {
    element_loss[i] = 0.0f;
    grad[i] = 0.0f;
    return;
  }

  float p = pred[i];
  float t = target[b];
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
