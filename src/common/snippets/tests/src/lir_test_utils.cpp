// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lir_test_utils.hpp"

#include "snippets/lowered/linear_ir_builder.hpp"
#include "snippets/utils.hpp"

using namespace ov::snippets::lowered;
using namespace ov::snippets::utils;
using namespace ov::snippets;

namespace ov {
namespace test {
namespace snippets {
LoweredPassTestsF::LoweredPassTestsF() : comparator(LIRComparator::no_default()) {
    Config lir_config;
    lir_config.m_manual_build_support = true;
    linear_ir = std::make_shared<LinearIR>(lir_config);
    linear_ir_ref = std::make_shared<LinearIR>(lir_config);

    comparator.enable(LIRComparator::NodesCmpValues::NODES);
    comparator.enable(LIRComparator::NodesCmpValues::CONST_VALUES);
    comparator.enable(LIRComparator::NodesCmpValues::RUNTIME_KEYS);
    comparator.enable(LIRComparator::NodesCmpValues::PRECISIONS);
    comparator.enable(LIRComparator::NodesCmpValues::ATTRIBUTES);
}

void LoweredPassTestsF::TearDown() {
    OPENVINO_ASSERT(linear_ir, "Test LIR is not initialized.");
    OPENVINO_ASSERT(linear_ir_ref, "Reference LIR is not initialized.");
    if (linear_ir_ref->get_ops().empty()) {
        linear_ir_ref = lowered::LinearIRBuilder().clone(linear_ir);
    }
    pipeline.run(*linear_ir);
    auto res = comparator.compare(linear_ir, linear_ir_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

ov::snippets::VectorDims get_default_subtensor() {
    static const VectorDims default_subtensor{PortDescriptor::ServiceDimensions::FULL_DIM,
                                              PortDescriptor::ServiceDimensions::FULL_DIM};
    return default_subtensor;
}

void init_expr_descriptors(const ov::snippets::lowered::ExpressionPtr& expr,
                           const std::vector<ov::snippets::VectorDims>& subtensors,
                           const std::vector<ov::snippets::VectorDims>& layouts) {
    const auto n_inputs = expr->get_input_count();
    const auto n_outputs = expr->get_output_count();
    OPENVINO_ASSERT(subtensors.empty() || subtensors.size() == n_inputs + n_outputs,
                    "subtensors vec size (",
                    subtensors.size(),
                    ") is not equal to n_inputs + n_outputs (",
                    n_inputs + n_outputs,
                    ")");
    OPENVINO_ASSERT(layouts.empty() || layouts.size() == n_inputs + n_outputs,
                    "layouts vec size (",
                    layouts.size(),
                    ") is not equal to n_inputs + n_outputs (",
                    n_inputs + n_outputs,
                    ")");

    auto update_expr_desc = [](const PortDescriptorPtr& expr_desc, const PortDescriptorPtr& new_desc) {
        expr_desc->set_shape(new_desc->get_shape());
        expr_desc->set_layout(new_desc->get_layout());
        expr_desc->set_subtensor(new_desc->get_subtensor());
    };

    const auto node = expr->get_node();
    for (size_t i = 0; i < n_inputs; ++i) {
        const auto& subtensor = subtensors.empty() ? get_default_subtensor() : subtensors[i];
        const auto& layout = layouts.empty() ? VectorDims{} : layouts[i];
        const auto desc = std::make_shared<PortDescriptor>(pshape_to_vdims(node->get_input_partial_shape(i)), subtensor, layout);
        PortDescriptorUtils::set_port_descriptor_ptr(node->input(i), desc);
        update_expr_desc(expr->get_input_port_descriptor(i), desc);
    }
    for (size_t i = 0; i < n_outputs; ++i) {
        const auto& subtensor = subtensors.empty() ? get_default_subtensor() : subtensors[i + n_inputs];
        const auto& layout = layouts.empty() ? VectorDims{} : layouts[i + n_inputs];
        const auto desc = std::make_shared<PortDescriptor>(pshape_to_vdims(node->get_output_partial_shape(i)), subtensor, layout);
        PortDescriptorUtils::set_port_descriptor_ptr(node->output(i), desc);
        update_expr_desc(expr->get_output_port_descriptor(i), desc);
    }
}

void create_and_add_unified_loop_info(const LinearIRPtr& linear_ir,
                                      size_t work_amount,
                                      size_t increment,
                                      const std::vector<LoopPort>& entries,
                                      const std::vector<LoopPort>& exits,
                                      bool set_default_handlers) {
    // Equal begin and end iterators are set to avoid expressions marking with new loop id
    create_and_add_unified_loop_info(linear_ir, linear_ir->begin(), linear_ir->begin(), work_amount, increment, entries, exits, set_default_handlers);
}

void create_and_add_unified_loop_info(const LinearIRPtr& linear_ir,
                                      ov::snippets::lowered::LinearIR::constExprIt loop_begin_pos,
                                      ov::snippets::lowered::LinearIR::constExprIt loop_end_pos,
                                      size_t work_amount,
                                      size_t increment,
                                      const std::vector<LoopPort>& entries,
                                      const std::vector<LoopPort>& exits,
                                      bool set_default_handlers) {
    const auto& loop_manager = linear_ir->get_loop_manager();
    loop_manager->mark_loop(loop_begin_pos, loop_end_pos, work_amount, increment, entries, exits, set_default_handlers);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
