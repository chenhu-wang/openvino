// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace pass {
using namespace lowered;

SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax_v1_m = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto softmax_v8_m = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>();
    auto softmax_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{softmax_v1_m, softmax_v8_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxDecomposition")
        auto softmax = m.get_match_root();

        const auto& pshape = softmax->get_input_partial_shape(0);
        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "SoftmaxDecomposition doesn't support dynamic ranks");
        const auto rank = pshape.size();

        size_t axis;
        if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            axis = ov::normalize_axis(softmax->get_friendly_name(), softmax_v8->get_axis(), rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
            axis = softmax_v1->get_axis();
        } else {
            OPENVINO_THROW("Unexpected node matched");
        }

        const auto& softmax_input = softmax->input_value(0);
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, axis);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, axis);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        OPENVINO_ASSERT(axis < rank, "Softmax has incorrect axis");
        std::vector<size_t> subtensor(rank, 1);
        for (size_t i = axis; i < rank; ++i)
            subtensor[i] = PortDescriptor::ServiceDimensions::FULL_DIM;

        PortDescriptorUtils::set_port_descriptor_ptr(reduce_max->input(0), std::make_shared<PortDescriptor>(reduce_max->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_max->output(0), std::make_shared<PortDescriptor>(reduce_max->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->input(0), std::make_shared<PortDescriptor>(reduce_sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->output(0), std::make_shared<PortDescriptor>(reduce_sum->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(power->input(0), std::make_shared<PortDescriptor>(power->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(power->output(0), std::make_shared<PortDescriptor>(power->output(0), subtensor));

        copy_runtime_info(softmax, {reduce_max, subtract, exp, reduce_sum, power, multiply});
        return ov::replace_node_update_name(softmax, multiply);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
