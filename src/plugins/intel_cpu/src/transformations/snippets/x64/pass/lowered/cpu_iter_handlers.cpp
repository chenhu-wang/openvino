// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_iter_handlers.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
using LinearIR = snippets::lowered::LinearIR;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

SetBrgemmBeta::SetBrgemmBeta(float beta) : snippets::lowered::pass::RangedPass(), m_beta(beta) {}

bool SetBrgemmBeta::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = expr_it->get();
        if (const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            brgemm->set_beta(m_beta);
        }
    }
    return true;
}

bool SetBrgemmBeta::can_be_merged(const std::shared_ptr<snippets::lowered::pass::PassBase>& other) {
    const auto casted_pass = ov::as_type_ptr<SetBrgemmBeta>(other);
    return casted_pass && m_beta == casted_pass->m_beta;
}
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov