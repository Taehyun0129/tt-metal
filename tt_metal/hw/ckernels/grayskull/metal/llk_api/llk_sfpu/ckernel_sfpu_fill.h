// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_fill(const float value) {
    // SFPU microcode
    vFloat fill_val = value;

    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = fill_val;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_fill_bitcast(const uint32_t value_bit_mask) {
    // SFPU microcode
    Converter c_value;
    c_value.u = value_bit_mask;
    vFloat fill_val = c_value.f;

    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = fill_val;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
