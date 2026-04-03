__kernel void adam_update(__global float *param, __global const float *grad,
                          __global float *m, __global float *v, const float lr,
                          const float beta1, const float beta2, const float eps,
                          const float beta1_t, const float beta2_t,
                          const unsigned int n) {
  int i = get_global_id(0);
  if (i >= n)
    return;

  float g = grad[i];
  float mi = beta1 * m[i] + (1.0f - beta1) * g;
  float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
  m[i] = mi;
  v[i] = vi;

  float m_hat = mi / (1.0f - beta1_t);
  float v_hat = vi / (1.0f - beta2_t);

  param[i] -= lr * m_hat / (sqrt(v_hat) + eps);
}
