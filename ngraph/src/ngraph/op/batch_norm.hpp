//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <memory>

#include "ngraph/deprecated.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API BatchNormInference : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"BatchNormInference", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                BatchNormInference() = default;
                /// \param input [., C, ...]
                /// \param gamma gamma scaling for normalized value. [C]
                /// \param beta bias added to the scaled normalized value [C]
                /// \param mean value for mean normalization [C]
                /// \param variance value for variance normalization [C]
                /// \param epsilon Avoids divsion by 0 if input has 0 variance
                BatchNormInference(const Output<Node>& input,
                                   const Output<Node>& gamma,
                                   const Output<Node>& beta,
                                   const Output<Node>& mean,
                                   const Output<Node>& variance,
                                   double epsilon);

                bool visit_attributes(AttributeVisitor& visitor) override;

                void validate_and_infer_types() override;

                double get_eps_value() const { return m_epsilon; }
                void set_eps_value(double epsilon) { m_epsilon = epsilon; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            private:
                static constexpr size_t INPUT_GAMMA = 0;
                static constexpr size_t INPUT_BETA = 1;
                static constexpr size_t INPUT_DATA = 2;
                static constexpr size_t INPUT_MEAN = 3;
                static constexpr size_t INPUT_VARIANCE = 4;

                double m_epsilon;
            };
        } // namespace v0
        using v0::BatchNormInference;
    }
}
