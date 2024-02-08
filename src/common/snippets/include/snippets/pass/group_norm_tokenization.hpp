// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface TokenizeGroupNormSnippets
 * @brief Tokenize GroupNormalization to a subgraph
 * @ingroup snippets
 */
class TokenizeGroupNormSnippets: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TokenizeGroupNormSnippets", "0");
    TokenizeGroupNormSnippets();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov