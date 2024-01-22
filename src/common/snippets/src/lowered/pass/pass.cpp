// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/pass.hpp"

#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

PassPipeline::PassPipeline() : m_pass_config(std::make_shared<PassConfig>()) {}
PassPipeline::PassPipeline(const std::shared_ptr<PassConfig>& pass_config) : m_pass_config(pass_config) {
    OPENVINO_ASSERT(m_pass_config != nullptr, "PassConfig is not initialized!");
}

void PassPipeline::register_pass(const snippets::pass::PassPosition& position, const std::shared_ptr<PassBase>& pass) {
    OPENVINO_ASSERT(pass != nullptr, "PassPipeline cannot register empty pass!");
    m_passes.insert(position.get_insert_position(m_passes), pass);
}

void PassPipeline::register_pass(const std::shared_ptr<PassBase>& pass) {
    OPENVINO_ASSERT(pass != nullptr, "PassPipeline cannot register empty pass!");
    m_passes.push_back(pass);
}

void PassPipeline::run(LinearIR& linear_ir) const {
    run(linear_ir, linear_ir.cbegin(), linear_ir.cend());
}

void PassPipeline::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) const {
    for (const auto& pass : m_passes) {
        OPENVINO_ASSERT(pass != nullptr, "PassPipeline has empty pass!");
        if (m_pass_config->is_disabled(pass->get_type_info())) {
            continue;
        }
        if (auto lir_pass = std::dynamic_pointer_cast<Pass>(pass)) {
            lir_pass->run(linear_ir);
        } else if (auto ranged_pass = std::dynamic_pointer_cast<RangedPass>(pass)) {
            ranged_pass->run(linear_ir, begin, end);
        } else {
            OPENVINO_THROW("Unexpected pass (", pass->get_type_info(), ") is registered in PassPipeline");
        }
    }
}

void PassPipeline::register_positioned_passes(const std::vector<PositionedPassLowered>& pos_passes) {
    for (const auto& pp : pos_passes)
        register_pass(pp.position, pp.pass);
}

PassPipeline PassPipeline::merge_pipelines(const PassPipeline& lhs, const PassPipeline& rhs) {
    OPENVINO_ASSERT(*lhs.get_pass_config() == *rhs.get_pass_config(), "2 passes with different PassConfigs can't be merged.");
    const auto& lhs_passes = lhs.get_passes();
    std::unordered_map<ov::DiscreteTypeInfo, std::shared_ptr<lowered::pass::PassBase>> passes_map;
    for (const auto& pass : lhs_passes) {
        passes_map[pass->get_type_info()] = pass;
    }

    auto merged_pipeline = lhs;
    for (const auto& pass : rhs.get_passes()) {
        auto lhs_pass_it = passes_map.find(pass->get_type_info());
        if (lhs_pass_it == passes_map.end()) {
            merged_pipeline.register_pass(pass);
        } else {
            OPENVINO_ASSERT(lhs_pass_it->second->can_be_merged(pass), "2 passes with type info ", pass->get_type_info(), " can't be merged.");
        }
    }
    return merged_pipeline;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
