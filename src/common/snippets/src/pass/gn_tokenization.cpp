// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/gn_tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"

#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::snippets::pass::TokenizeGNSnippets::TokenizeGNSnippets() {
    MATCHER_SCOPE(TokenizeGNSnippets);

    auto group_norm_pattern = ov::pass::pattern::wrap_type<ov::op::v12::GroupNormalization>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::pass::TokenizeGNSnippets")
        auto group_norm_node = ov::as_type_ptr<ov::op::v12::GroupNormalization>(m.get_match_root());
        if (group_norm_node->is_dynamic() ||
            TokenizeSnippets::get_supported_element_types().count(group_norm_node->get_element_type()) == 0)
            return false;

        auto subgraph = op::Subgraph::wrap_node_as_subgraph(group_norm_node);
        subgraph->get_rt_info()["originalLayersNames"] = group_norm_node->get_friendly_name();
        ov::replace_node(group_norm_node, subgraph);
        op::update_out_tensor_name(subgraph);

        // mark the Subgraph as Completed to not allow Snippets to include any nodes into the GN Subgraph in common Tokenization
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_norm_pattern, matcher_name);
    register_matcher(m, callback);
}
