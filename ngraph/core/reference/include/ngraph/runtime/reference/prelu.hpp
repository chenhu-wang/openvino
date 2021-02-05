//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <cstddef>
#include <cassert>
#include <ngraph/op/util/attr_types.hpp>
#include <ngraph/shape.hpp>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void prelu(const T* arg,
                       const T* slope,
                       T* out,
                       const Shape& arg_shape,
                       const Shape& slope_shape)
            {
                int cnt = 0;
                for (size_t i = 0; i < shape_size(arg_shape); ++i)
                {
                    out[i] =
                        arg[i] < T(0) ? T(arg[i] * slope[cnt++ % shape_size(slope_shape)]) : arg[i];
                }
                // assert(arg_shape.size() > 2);
                // size_t batch_size = arg_shape[0] * arg_shape[1];
                // size_t sptial_size = shape_size(arg_shape) / batch_size;
                // for (int n = 0; n < batch_size; ++n)
                // {
                //     T slope_n = shape_size(slope_shape) == 1 ? slope[0] : slope[n % shape_size(slope_shape)];
                //     for (int sp = 0; sp < sptial_size; ++sp)
                //     {
                //         size_t inx = n * sptial_size + sp;
                //         out[inx] =
                //             arg[inx] < T(0) ? T(arg[inx] * slope_n) : arg[inx];
                //     }
                // }
            }
        }
    }
}
