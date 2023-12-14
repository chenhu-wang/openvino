// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/op/loop.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
class UpdateMemoryAccessOps : public pass::RangedPass {
public:
    UpdateMemoryAccessOps(size_t count);
    OPENVINO_RTTI("UpdateMemoryAccessOps", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_count;
};

class SetFillOffset : public pass::RangedPass {
public:
    SetFillOffset(size_t offset);
    OPENVINO_RTTI("SetFillOffset", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_offset;
};

class TransformInnerSplitLoop : public pass::RangedPass {
public:
    TransformInnerSplitLoop(size_t tail_size);
    OPENVINO_RTTI("TransformInnerSplitLoop", "RangedPass")
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_tail_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov