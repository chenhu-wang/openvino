// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#include <cstddef>

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
template <typename T>
void equal(const T* arg0,
           const T* arg1,
           char* out,
           size_t count)  // TODO: using char for bool, is this right?
{
    for (size_t i = 0; i < count; i++) {
        out[i] = arg0[i] == arg1[i];
    }
}

template <typename T, typename U>
void equal(const T* arg0,
           const T* arg1,
           U* out,
           const Shape& arg0_shape,
           const Shape& arg1_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> U {
        return static_cast<U>(x == y);
    });
}
}  // namespace reference
}  // namespace ov

#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif