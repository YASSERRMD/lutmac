#pragma once

#include <cstddef>
#include <vector>

namespace lutmac {

/**
 * Fast Walsh-Hadamard Transform (FWHT) for size 256.
 * Performs in-place transformation: x -> H*x.
 *
 * Optimized for NEON/AVX2.
 */
void fwht_256(float *data);

/**
 * Apply FWHT to all 256-element blocks in a buffer.
 */
void apply_hadamard_rotation(float *data, size_t n);

} // namespace lutmac
