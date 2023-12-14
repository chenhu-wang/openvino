// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/reduce_to_snippets_reduce.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reduce.hpp"

namespace ov {
namespace snippets {
namespace pass {
using namespace lowered;
snippets::pass::ReduceToSnippetsReduce::ReduceToSnippetsReduce() {
    MATCHER_SCOPE(ReduceToSnippetsReduce);
    auto reduce_pattern = ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum, ov::op::v1::ReduceMax>();

    auto callback = [](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ReduceToSnippetsReduce")
        auto reduce = m.get_match_root();
        const auto& reduce_base = ov::as_type_ptr<ov::op::util::ArithmeticReductionKeepDims>(reduce);
        OPENVINO_ASSERT(reduce_base, "Failed to cast Reduce operation to ArithmeticReductionKeepDims");
        const auto& axis_constant = ov::as_type_ptr<ov::op::v0::Constant>(reduce->get_input_node_shared_ptr(1));
        // Note: we do not check the Constant value here. If the Reduce was tokenized, then we assume that it is supported
        OPENVINO_ASSERT(reduce_base->get_keep_dims() && axis_constant, "Unspported Reduce was tokenized by Snippets");
        const auto& data_input = reduce->get_input_source_output(0);
        const auto reduce_rank = reduce->get_input_partial_shape(0).rank();
        const auto axis = ov::normalize_axis(reduce->get_friendly_name(), axis_constant->cast_vector<int32_t>(1)[0], reduce_rank);

        std::shared_ptr<snippets::op::ReduceBase> snippets_reduce = nullptr;
        if (ov::is_type<ov::op::v1::ReduceSum>(reduce))
            snippets_reduce = std::make_shared<snippets::op::ReduceSum>(data_input, axis);
        else if (ov::is_type<ov::op::v1::ReduceMax>(reduce))
            snippets_reduce = std::make_shared<snippets::op::ReduceMax>(data_input, axis);
        else
            OPENVINO_THROW("Reduce ", reduce, " can't be converted to snippets opset.");

        std::vector<size_t> subtensor(reduce_rank.get_length(), 1);
        for (auto i = axis; i < reduce_rank.get_length(); ++i)
            subtensor[i] = PortDescriptor::ServiceDimensions::FULL_DIM;
        PortDescriptorUtils::set_port_descriptor_ptr(snippets_reduce->input(0), std::make_shared<PortDescriptor>(snippets_reduce->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(snippets_reduce->output(0), std::make_shared<PortDescriptor>(snippets_reduce->output(0), subtensor));

        ov::replace_node(reduce, snippets_reduce);
        snippets_reduce->set_friendly_name(reduce->get_friendly_name());
        ov::copy_runtime_info(reduce, snippets_reduce);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reduce_pattern, matcher_name);
    register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ov