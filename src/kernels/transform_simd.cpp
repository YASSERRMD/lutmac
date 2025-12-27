#include "lutmac/transform.hpp"
#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace lutmac {

// Helper: Scalar pass for small strides
inline void fwht_pass_scalar(float *data, size_t h) {
  for (size_t i = 0; i < 256; i += 2 * h) {
    for (size_t j = i; j < i + h; ++j) {
      float x = data[j];
      float y = data[j + h];
      data[j] = x + y;
      data[j + h] = x - y;
    }
  }
}

// Optimized FWHT-256
void fwht_256(float *data) {
  // Strides 1, 2: Scalar (auto-vectorization opportunity)
  // Can be optimized manually with intrinsics but scalar is safe/clean for now
  fwht_pass_scalar(data, 1);
  fwht_pass_scalar(data, 2);

  // Strides 4, 8, 16, 32, 64, 128: Fully vectorized
  for (size_t h = 4; h < 256; h <<= 1) {
    for (size_t i = 0; i < 256; i += 2 * h) {
      // j loop processes 'h' elements. h >= 4.
      for (size_t j = 0; j < h; j += 4) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t vm = vld1q_f32(data + i + j);
        float32x4_t vp = vld1q_f32(data + i + h + j);

        float32x4_t vsum = vaddq_f32(vm, vp);
        float32x4_t vdiff = vsubq_f32(vm, vp);

        vst1q_f32(data + i + j, vsum);
        vst1q_f32(data + i + h + j, vdiff);
#else
        // Fallback scalar within blocks
        for (size_t k = 0; k < 4; ++k) {
          float x = data[i + j + k];
          float y = data[i + h + j + k];
          data[i + j + k] = x + y;
          data[i + h + j + k] = x - y;
        }
#endif
      }
    }
  }

  // Normalization?
  // Standard FWHT scales by Factor? Usually orthogonal Hadamard is normalized
  // by 1/sqrt(N). FWHT typically computes N*result or unscaled. User requested
  // "Rotation". If we rotation W and X, we get Y_rot. If H is unscaled
  // (elements 1/-1), H H^T = 256 I. W' = W H. X' = X H. X' (W')^T = X H H^T W^T
  // = 256 X I W^T = 256 X W^T. So result is scaled by 256. We should scale
  // inputs by 1/sqrt(256) = 1/16? Or scale result by 1/256? Doing it at
  // inference time is cheaper (1 mul per result). Or scale weights by 1/16 and
  // activations by 1/16.

  // Let's perform scaling by 1/16 (0.0625) here to preserve range stability.
  // This makes the transform orthogonal (unitary).
  // 256 / 4 = 64 iters.
  float inv_scale = 1.0f / 16.0f;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t vscale = vdupq_n_f32(inv_scale);
  for (size_t i = 0; i < 256; i += 4) {
    vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), vscale));
  }
#else
  for (size_t i = 0; i < 256; ++i) {
    data[i] *= inv_scale;
  }
#endif
}

void apply_hadamard_rotation(float *data, size_t n) {
  // Pad to 256 if needed?
  // User task implies models are divisible by 256 (hidden 1536 etc).
  // We assume n is multiple of 256.
  for (size_t i = 0; i < n; i += 256) {
    if (i + 256 <= n) {
      fwht_256(data + i);
    }
  }
}

} // namespace lutmac
