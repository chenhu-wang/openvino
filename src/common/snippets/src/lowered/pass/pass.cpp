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

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
