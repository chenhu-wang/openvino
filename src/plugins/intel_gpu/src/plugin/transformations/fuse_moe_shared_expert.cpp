// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_shared_expert.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

#define MOE_COMPRESSED_SHARED_EXPERT_PATTERN(SUFFIX)\
    auto gemm3_compressed_weights_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches_any({ov::element::u4, ov::element::u8, ov::element::i4, ov::element::i8}));\
    auto gemm3_zp_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches_any({ov::element::u4, ov::element::u8, ov::element::i4, ov::element::i8}));\
    \
    auto gemm3_weight_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({gemm3_compressed_weights_m_##SUFFIX}, type_matches(ov::element::f16));\
    auto gemm3_zp_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({gemm3_zp_m_##SUFFIX}, type_matches(ov::element::f16));\
    auto gemm3_sub_m_##SUFFIX = wrap_type<ov::op::v1::Subtract>({gemm3_weight_convert_m_##SUFFIX, gemm3_zp_convert_m_##SUFFIX});\
    \
    auto gemm3_scale_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));\
    /* Asymmetric: Convert -> Subtract(zp) -> Multiply(scale) */\
    auto gemm3_mul_asym_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({gemm3_sub_m_##SUFFIX, gemm3_scale_m_##SUFFIX});\
    /* Symmetric: Convert -> Multiply(scale), no Subtract */\
    auto gemm3_mul_sym_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({gemm3_weight_convert_m_##SUFFIX, gemm3_scale_m_##SUFFIX});\
    auto gemm3_mul_m_##SUFFIX = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{gemm3_mul_asym_m_##SUFFIX, gemm3_mul_sym_m_##SUFFIX});\
    \
    auto gemm3_reshape_ungroup_##SUFFIX = [](const ov::Output<ov::Node>& output) {\
        auto in_ps = output.get_node()->get_input_partial_shape(0);\
        auto out_ps = output.get_node()->get_output_partial_shape(0);\
        return in_ps.rank().is_static() && out_ps.rank().is_static() &&\
        ((in_ps.size() == 4 && out_ps.size() == 3) || (in_ps.size() == 3 && out_ps.size() == 2));\
    };\
    \
    auto gemm3_reshape_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto gemm3_reshape_m_##SUFFIX = optional<ov::op::v1::Reshape>({gemm3_mul_m_##SUFFIX, gemm3_reshape_const_m_##SUFFIX}, gemm3_reshape_ungroup_##SUFFIX);\
    \
    auto gemm3_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({gemm3_reshape_m_##SUFFIX}, type_matches(ov::element::f32));

FuseMOESharedExpert::FuseMOESharedExpert() {
    using namespace ov::pass::pattern;

    // Match the MOE node (GEMM3_SWIGLU type, 6 inputs: hidden, routing, topk, gate, up, down)
    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();
    auto gate_weight_m = any_input();
    auto up_weight_m = any_input();
    auto down_weight_m = any_input();

    auto moe_m = wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, gate_weight_m, up_weight_m, down_weight_m},
                                                  [](const ov::Output<ov::Node>& output) {
                                                      auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
                                                      return moe && moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
                                                  });

    // Shared expert subgraph:
    //   shared_gate = MatMul(shared_hidden, shared_gate_weight)
    //   shared_swish = Swish(shared_gate)
    //   shared_up   = MatMul(shared_hidden, shared_up_weight)
    //   shared_mul  = Mul(shared_swish, shared_up)
    //   shared_down = MatMul(shared_mul, shared_down_weight)
    //   Optional gating: sigmoid(MatMul(shared_hidden, gate_gate_weight)) * shared_down
    //   Optional reshape before Add
    MOE_COMPRESSED_SHARED_EXPERT_PATTERN(gate)
    MOE_COMPRESSED_SHARED_EXPERT_PATTERN(up)
    MOE_COMPRESSED_SHARED_EXPERT_PATTERN(down)
    auto shared_hidden_states_m = any_input();
    auto shared_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, gemm3_convert_m_gate});
    auto shared_swish_m = wrap_type<ov::op::v4::Swish>({shared_gate_m});
    auto shared_up_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, gemm3_convert_m_up});
    // Multiply is commutative: handle both input orders
    auto shared_mul_m_1 = wrap_type<ov::op::v1::Multiply>({shared_swish_m, shared_up_m});
    auto shared_mul_m_2 = wrap_type<ov::op::v1::Multiply>({shared_up_m, shared_swish_m});
    auto shared_mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_mul_m_1, shared_mul_m_2});
    auto shared_down_m = wrap_type<ov::op::v0::MatMul>({shared_mul_m, gemm3_convert_m_down});

    // Optional sigmoid gating: sigmoid(MatMul(hidden, gate_gate)) * down
    auto shared_gate_gate_wei_m = any_input();
    auto shared_gate_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, shared_gate_gate_wei_m});
    auto shared_gate_sigmoid_m = wrap_type<ov::op::v0::Sigmoid>({shared_gate_gate_m});
    // Multiply is commutative: handle both input orders
    auto shared_expert_gated_m_1 = wrap_type<ov::op::v1::Multiply>({shared_gate_sigmoid_m, shared_down_m});
    auto shared_expert_gated_m_2 = wrap_type<ov::op::v1::Multiply>({shared_down_m, shared_gate_sigmoid_m});
    auto shared_expert_gated_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_expert_gated_m_1, shared_expert_gated_m_2});
    auto shared_expert_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_down_m, shared_expert_gated_m});
    auto shared_expert_reshaped_m = optional<ov::op::v1::Reshape>({shared_expert_m, any_input()});

    // Root: Add(MOE, SharedExpert) or Add(SharedExpert, MOE)
    auto add_1 = wrap_type<ov::op::v1::Add>({moe_m, shared_expert_reshaped_m});
    auto add_2 = wrap_type<ov::op::v1::Add>({shared_expert_reshaped_m, moe_m});
    auto root = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add_1, add_2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto root_node = pattern_map.at(root).get_node_shared_ptr();
        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(pattern_map.at(moe_m).get_node_shared_ptr());
        if (!moe || transformation_callback(root_node)) {
            return false;
        }

        // Build new MOE with shared expert weights as additional inputs:
        //   {hidden, routing, topk, gate, up, down, sh_gate, sh_up, sh_down, gate_gate}
        OutputVector new_inputs = {
            moe->input_value(0),                   // hidden_states
            moe->input_value(1),                   // routing_weights
            moe->input_value(2),                   // topk_indices
            moe->input_value(3),                   // gate weight (decompressed)
            moe->input_value(4),                   // up weight (decompressed)
            moe->input_value(5),                   // down weight (decompressed)
            pattern_map.at(gemm3_convert_m_gate),  // shared gate weight
            pattern_map.at(gemm3_convert_m_up),    // shared up weight
            pattern_map.at(gemm3_convert_m_down),  // shared down weight
        };

        // Use shared_gate_sigmoid_m to reliably detect whether the gating branch matched.
        // shared_gate_gate_wei_m is any_input() and may be spuriously bound even when
        // the Or pattern took the non-gating (shared_down_m) branch.
        bool has_gating = pattern_map.count(shared_gate_sigmoid_m) > 0;
        if (has_gating) {
            new_inputs.push_back(pattern_map.at(shared_gate_gate_wei_m));
        } else {
            // Models without gate_gate: insert a dummy so input count is always 10
            size_t hidden_size = moe->get_output_partial_shape(0).rbegin()->get_length();
            new_inputs.push_back(
                ov::op::v0::Constant::create(ov::element::f16, ov::Shape{hidden_size, 1}, std::vector<float>(hidden_size, 0.0f)));
        }

        auto new_moe = std::make_shared<ov::op::internal::MOE>(new_inputs, moe->get_config());
        new_moe->set_friendly_name(root_node->get_friendly_name());
        ov::copy_runtime_info({moe, root_node}, new_moe);
        ov::replace_node(root_node, new_moe);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "FuseMOESharedExpert");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
