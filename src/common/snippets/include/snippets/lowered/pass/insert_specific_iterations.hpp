// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertSpecificIterations
 * @brief Inserts separate loop bodies for first/last iterations if needed.
 * Also calls previously registered SpecificIterationHandlers for the inserted bodies and the main body.
 * @ingroup snippets
 */
class InsertSpecificIterations : public RangedPass {
public:
    OPENVINO_RTTI("InsertSpecificIterations", "RangedPass")
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

    static LinearIR::container copy_loop(const LinearIR& linear_ir, const size_t loop_id);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
