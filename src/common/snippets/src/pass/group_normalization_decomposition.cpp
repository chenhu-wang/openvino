// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/group_normalization_decomposition.hpp"

#include "openvino/op/group_normalization.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/snippets_isa.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov {
namespace snippets {
namespace pass {
using namespace lowered;

// groupNorm -> reshape + mvn + reshape + mul + add,
// where mvn = (x - mean) / Sqrt(ReduceMean((x - mean) ^ 2) + eps),
// where mean = ReduceMean(x, axes)
GroupNormalizationDecomposition::GroupNormalizationDecomposition() {
    MATCHER_SCOPE(GroupNormalizationDecomposition);
    auto group_norm_pattern = ov::pass::pattern::wrap_type<ov::op::v12::GroupNormalization>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::pass::GroupNormalizationDecomposition")
        auto group_norm_node = ov::as_type_ptr<ov::op::v12::GroupNormalization>(m.get_match_root());

        const auto data = group_norm_node->input_value(0);
        const auto scale = group_norm_node->input_value(1);
        const auto bias = group_norm_node->input_value(2);

        const auto num_groups = static_cast<size_t>(group_norm_node->get_num_groups());
        const float eps = static_cast<float>(group_norm_node->get_epsilon());

        // enforce 4 rank
        ////////////collapse to reduce lastDim///////////
        // reshape [N, C, spatial] to [N, group, 1, (C / group) * spatial]
        const auto orig_shape = group_norm_node->get_input_partial_shape(0);
        size_t orig_rank = orig_shape.rank().get_length();
        size_t group_rank = 4;
        std::vector<Dimension> group_dims(group_rank);
        group_dims[0] = orig_shape[0];
        group_dims[1] = Dimension(num_groups);
        group_dims[2] = Dimension(1);
        group_dims[3] = Dimension(orig_shape[1] / num_groups);
        Dimension spatial_dim = 1;
        for (size_t i = 2; i < orig_rank; ++i) {
            spatial_dim = spatial_dim * orig_shape[i];
        }
        group_dims[3] = group_dims[3] * spatial_dim;
        ov::PartialShape group_shape(group_dims);
        const auto reshaped_node = std::make_shared<ov::snippets::op::Reshape>(data, group_shape);

        // reduceSum on dimension [C / group * spatial]
        std::vector<int64_t> axis(1, 3);
        auto axis_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{axis.size()}, axis);
        const auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(reshaped_node, axis_node, true);

        // reduceMean
        auto group_shape_static = group_shape.to_shape();
        int64_t group_size = group_shape_static[3];
        auto group_size_node = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{1}, std::vector<int64_t>{group_size});

        const auto group_size_inv = std::make_shared<ov::snippets::op::PowerStatic>(group_size_node, -1.f);
        const auto reduce_mean = std::make_shared<ov::op::v1::Multiply>(reduce_sum, group_size_inv);

        // x - mean
        auto mean_norm = std::make_shared<ov::op::v1::Subtract>(reshaped_node, reduce_mean);
        // (x - mean) ^ 2
        auto sqr_const = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{1}, std::vector<int64_t>{2});
        auto sqr = std::make_shared<ov::op::v1::Power>(mean_norm, sqr_const);
        // reduceSum((x - mean) ^ 2)
        auto mean_sum_variance = std::make_shared<ov::op::v1::ReduceSum>(sqr, axis_node, true);
        // reduceMean((x - mean) ^ 2)
        //
        // auto group_shape_static1 = group_shape.to_shape();
        // int64_t group_size1 = group_shape_static1[3];
        // auto group_size_node1 = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{1}, std::vector<int64_t>{group_size1});
        // const auto group_size_inv1 = std::make_shared<ov::snippets::op::PowerStatic>(group_size_node1, -1.f);
        //
        auto reduce_mean_variance = std::make_shared<ov::op::v1::Multiply>(mean_sum_variance, group_size_inv);
        // auto reduce_mean_variance = std::make_shared<ov::op::v1::Multiply>(mean_sum_variance, group_size_inv1);
        // reduceMean((x - mean) ^ 2) + eps
        auto eps_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{eps});
        auto eps_add = std::make_shared<ov::op::v1::Add>(reduce_mean_variance, eps_node);
        // variance = sqrt( reducemean( (x - mean) ^ 2 ) + eps )
        auto variance = std::make_shared<ov::op::v0::Sqrt>(eps_add);

        // div variance
        const auto variance_inv = std::make_shared<ov::snippets::op::PowerStatic>(variance, -1.f);
        auto mvn = std::make_shared<ov::op::v1::Multiply>(mean_norm, variance_inv);

        // reshape mvn to [N, group, C / group, spatial]
        size_t group1_rank = 4;
        std::vector<Dimension> group1_dims(group1_rank);
        group1_dims[0] = group_dims[0];
        group1_dims[1] = group_dims[1];
        group1_dims[2] = Dimension(orig_shape[1] / num_groups);
        group1_dims[3] = spatial_dim;
        ov::PartialShape group1_shape(group1_dims);
        const auto mvn_reshaped = std::make_shared<ov::snippets::op::Reshape>(mvn, group1_shape);

        // reshape scale and bias to [1, group, C / group, 1]
        std::vector<Dimension> scale_bias_dims(group1_rank, Dimension(1));
        scale_bias_dims[1] = group1_dims[1];
        scale_bias_dims[2] = group1_dims[2];
        ov::PartialShape scale_bias_shape(scale_bias_dims);
        const auto reshape_scale = std::make_shared<ov::snippets::op::Reshape>(scale, scale_bias_shape);
        const auto reshape_bias = std::make_shared<ov::snippets::op::Reshape>(bias, scale_bias_shape);

        // scaled
        auto scaled_node = std::make_shared<ov::op::v1::Multiply>(mvn_reshaped, reshape_scale);
        // scaled_node[2,5,2,8,8] scaled_node[1,5,2,1,1] -> result[2,5,2,8,8]
        auto biased_node = std::make_shared<ov::op::v1::Add>(scaled_node, reshape_bias);

        // reshape_back [N, group, C / group, spatial] to [N, C, spatial]
        const auto reshape_back_node = std::make_shared<ov::snippets::op::Reshape>(biased_node, orig_shape);
        // const auto reshape_back_node = std::make_shared<ov::snippets::op::Reshape>(mvn, orig_shape);

        std::vector<size_t> subtensor(group_rank, 1);
        subtensor[2] = PortDescriptor::ServiceDimensions::FULL_DIM;
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->input(0), std::make_shared<PortDescriptor>(reduce_sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->output(0), std::make_shared<PortDescriptor>(reduce_sum->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(mean_sum_variance->input(0), std::make_shared<PortDescriptor>(mean_sum_variance->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(mean_sum_variance->output(0), std::make_shared<PortDescriptor>(mean_sum_variance->output(0), subtensor));

        return ov::replace_node_update_name(group_norm_node, reshape_back_node);
        // return ov::replace_node_update_name(group_norm_node, reshape_back_node);
        // 4 rank end
/*
        ////////////collapse to reduce lastDim///////////
        // reshape [N, C, spatial] to [N, group, (C / group) * spatial]
        const auto orig_shape = group_norm_node->get_input_partial_shape(0);
        size_t orig_rank = orig_shape.rank().get_length();
        // size_t group_rank = 3;
        // std::vector<Dimension> group_dims(group_rank);
        // group_dims[0] = orig_shape[0];
        // group_dims[1] = Dimension(num_groups);
        // group_dims[2] = Dimension(orig_shape[1] / num_groups);
        // Dimension spatial_dim = 1;
        // for (size_t i = 2; i < orig_rank; ++i) {
        //     spatial_dim = spatial_dim * orig_shape[i];
        // }
        // group_dims[2] = group_dims[2] * spatial_dim;
        // ov::PartialShape group_shape(group_dims);
        // const auto reshaped_node = std::make_shared<ov::snippets::op::Reshape>(data, group_shape);

        // for same rank, reshape [N, C, spatial] to [N, group, 1, (C / group) * spatial]
        size_t group_rank = 4;
        std::vector<Dimension> group_dims(group_rank);
        group_dims[0] = orig_shape[0];
        group_dims[1] = Dimension(num_groups);
        group_dims[2] = Dimension(1);
        group_dims[3] = Dimension(orig_shape[1] / num_groups);
        Dimension spatial_dim = 1;
        for (size_t i = 2; i < orig_rank; ++i) {
            spatial_dim = spatial_dim * orig_shape[i];
        }
        group_dims[3] = group_dims[3] * spatial_dim;
        ov::PartialShape group_shape(group_dims);
        const auto reshaped_node = std::make_shared<ov::snippets::op::Reshape>(data, group_shape);
        //

        // reduceSum on dimension [C / group, spatial]
        // std::vector<int64_t> axis(1, 2);
        std::vector<int64_t> axis(1, 3);
        auto axis_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{axis.size()}, axis);
        const auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(reshaped_node, axis_node, true);

        // reduceMean
        auto group_shape_static = group_shape.to_shape();
        int64_t group_size = group_shape_static[2];
        auto group_size_node = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{}, std::vector<int64_t>{group_size});
        const auto group_size_inv = std::make_shared<ov::snippets::op::PowerStatic>(group_size_node, -1.f);
        const auto reduce_mean = std::make_shared<ov::op::v1::Multiply>(reduce_sum, group_size_inv);

        // x - mean
        auto mean_norm = std::make_shared<ov::op::v1::Subtract>(reshaped_node, reduce_mean);
        // (x - mean) ^ 2
        auto sqr_const = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{1}, std::vector<int64_t>{2});
        auto sqr = std::make_shared<ov::op::v1::Power>(mean_norm, sqr_const);
        // reduceSum((x - mean) ^ 2)
        auto mean_sum_variance = std::make_shared<ov::op::v1::ReduceSum>(sqr, axis_node, true);
        // reduceMean((x - mean) ^ 2)
        auto reduce_mean_variance = std::make_shared<ov::op::v1::Multiply>(mean_sum_variance, group_size_inv);
        // reduceMean((x - mean) ^ 2) + eps
        auto eps_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{eps});
        auto eps_add = std::make_shared<ov::op::v1::Add>(reduce_mean_variance, eps_node);
        // variance = sqrt( reducemean( (x - mean) ^ 2 ) + eps )
        auto variance = std::make_shared<ov::op::v0::Sqrt>(eps_add);

        // div variance
        const auto variance_inv = std::make_shared<ov::snippets::op::PowerStatic>(variance, -1.f);
        auto mvn = std::make_shared<ov::op::v1::Multiply>(mean_norm, variance_inv);

        // reshape mvn to [N, group, C / group, spatial]
        size_t group1_rank = 4;
        std::vector<Dimension> group1_dims(group1_rank);
        group1_dims[0] = group_dims[0];
        group1_dims[1] = group_dims[1];
        group1_dims[2] = Dimension(orig_shape[1] / num_groups);
        group1_dims[3] = spatial_dim;
        ov::PartialShape group1_shape(group1_dims);
        const auto mvn_reshaped = std::make_shared<ov::snippets::op::Reshape>(mvn, group1_shape);

        // reshape scale and bias to [1, group, C / group, 1]
        std::vector<Dimension> scale_bias_dims(group1_rank, Dimension(1));
        scale_bias_dims[1] = group1_dims[1];
        scale_bias_dims[2] = group1_dims[2];
        ov::PartialShape scale_bias_shape(scale_bias_dims);
        const auto reshape_scale = std::make_shared<ov::snippets::op::Reshape>(scale, scale_bias_shape);
        const auto reshape_bias = std::make_shared<ov::snippets::op::Reshape>(bias, scale_bias_shape);

        // scaled
        auto scaled_node = std::make_shared<ov::op::v1::Multiply>(mvn_reshaped, reshape_scale);
        // scaled_node[2,5,2,8,8] scaled_node[1,5,2,1,1] -> result[2,5,2,8,8]
        auto biased_node = std::make_shared<ov::op::v1::Add>(scaled_node, reshape_bias);

        // reshape_back [N, group, C / group, spatial] to [N, C, spatial]
        const auto reshape_back_node = std::make_shared<ov::snippets::op::Reshape>(biased_node, orig_shape);

        std::vector<size_t> subtensor(group_rank, 1);
        subtensor[2] = PortDescriptor::ServiceDimensions::FULL_DIM;
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->input(0), std::make_shared<PortDescriptor>(reduce_sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->output(0), std::make_shared<PortDescriptor>(reduce_sum->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(mean_sum_variance->input(0), std::make_shared<PortDescriptor>(mean_sum_variance->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(mean_sum_variance->output(0), std::make_shared<PortDescriptor>(mean_sum_variance->output(0), subtensor));

        std::cout << "GN_decompose" << std::endl;
        ////////////collapse to reduce lastDim end///////////
/**/

        // reshape [N, C, spatial] to [N, group, C / group, spatial]
        // const auto orig_shape = group_norm_node->get_input_partial_shape(0);
        // size_t orig_rank = orig_shape.rank().get_length();
        // size_t group_rank = orig_rank + 1;
        // std::vector<Dimension> group_dims(group_rank);
        // group_dims[0] = orig_shape[0];
        // group_dims[1] = Dimension(num_groups);
        // OPENVINO_ASSERT(orig_shape.is_static(), "Snippets only support static shape GroupNormalization");
        // group_dims[2] = Dimension(orig_shape[1] / num_groups);
        // for (size_t i = 3; i < group_rank; ++i) {
        //     group_dims[i] = orig_shape[i - 1];
        // }
        // ov::PartialShape group_shape(group_dims);
        // const auto reshaped_node = std::make_shared<ov::snippets::op::Reshape>(data, group_shape);

        // // reduceSum on dimension [C / group, spatial]
        // int64_t axis_start = 2;
        // std::vector<int64_t> axis(group_rank - axis_start);
        // std::iota(axis.begin(), axis.end(), axis_start); // axis:[2, 3, 4...]
        // auto axis_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{axis.size()}, axis);
        // const auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(reshaped_node, axis_node, true);

        // // reduceMean
        // auto group_shape_static = group_shape.to_shape();
        // int64_t group_size = std::accumulate(group_shape_static.begin() + axis_start, group_shape_static.end(), 1, std::multiplies<int64_t>());
        // auto group_size_node = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{}, std::vector<int64_t>{group_size});
        // // auto group_size_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<int64_t>{group_size});
        // const auto group_size_inv = std::make_shared<ov::snippets::op::PowerStatic>(group_size_node, -1.f);
        // // reduceSum convert to f32
        // const auto reduce_mean = std::make_shared<ov::op::v1::Multiply>(reduce_sum, group_size_inv);

        // // x - mean
        // // mean_norm to float
        // auto mean_norm = std::make_shared<ov::op::v1::Subtract>(reshaped_node, reduce_mean);
        // // (x - mean) ^ 2
        // auto sqr_const = std::make_shared<ov::op::v0::Constant>(data.get_element_type(), Shape{1}, std::vector<int64_t>{2});
        // auto sqr = std::make_shared<ov::op::v1::Power>(mean_norm, sqr_const);
        // // reduceSum((x - mean) ^ 2)
        // auto mean_sum_variance = std::make_shared<ov::op::v1::ReduceSum>(sqr, axis_node, true);
        // // reduceMean((x - mean) ^ 2)
        // auto reduce_mean_variance = std::make_shared<ov::op::v1::Multiply>(mean_sum_variance, group_size_inv);
        // // reduceMean((x - mean) ^ 2) + eps
        // auto eps_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{eps});
        // auto eps_add = std::make_shared<ov::op::v1::Add>(reduce_mean_variance, eps_node);
        // // variance = sqrt( reducemean( (x - mean) ^ 2 ) + eps )
        // auto variance = std::make_shared<ov::op::v0::Sqrt>(eps_add);

        // // ( (x - mean) / variance) * scale + bias
        // // reshape scale and bias
        // std::vector<Dimension> scale_bias_dims(group_rank, Dimension(1));
        // scale_bias_dims[1] = group_shape[1];
        // scale_bias_dims[2] = group_shape[2];
        // ov::PartialShape scale_bias_shape(scale_bias_dims);
        // const auto reshape_scale = std::make_shared<ov::snippets::op::Reshape>(scale, scale_bias_shape);
        // const auto reshape_bias = std::make_shared<ov::snippets::op::Reshape>(bias, scale_bias_shape);

        // // e.g, orig[2,10,8,8], reshaped[2,5,2,8,8],
        // // variance_inv[2,5,1,1,1] reshape_scale[1,5,2,1,1] -> result[2,5,2,1,1]
        // const auto variance_inv = std::make_shared<ov::snippets::op::PowerStatic>(variance, -1.f);
        // auto scaled_variance_inv = std::make_shared<ov::op::v1::Multiply>(variance_inv, reshape_scale);
        // // this enable MulAddToFMA afterwards
        // // mean_norm[2,5,2,8,8] scaled_variance_inv[2,5,2,1,1] -> result[2,5,2,8,8]
        // auto scaled_node = std::make_shared<ov::op::v1::Multiply>(mean_norm, scaled_variance_inv);
        // // scaled_node[2,5,2,8,8] scaled_node[1,5,2,1,1] -> result[2,5,2,8,8]
        // auto biased_node = std::make_shared<ov::op::v1::Add>(scaled_node, reshape_bias);

        // // reshape_back [N, group, C / group, spatial] to [N, C, spatial]
        // const auto reshape_back_node = std::make_shared<ov::snippets::op::Reshape>(biased_node, orig_shape);

        // std::vector<size_t> subtensor(group_rank, 1);
        // for (size_t i = axis_start; i < group_rank; ++i)
        //     subtensor[i] = PortDescriptor::ServiceDimensions::FULL_DIM;
    // PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->input(0), std::make_shared<PortDescriptor>(reduce_sum->input(0), subtensor));
    // PortDescriptorUtils::set_port_descriptor_ptr(reduce_sum->output(0), std::make_shared<PortDescriptor>(reduce_sum->output(0), subtensor));
    // PortDescriptorUtils::set_port_descriptor_ptr(mean_sum_variance->input(0), std::make_shared<PortDescriptor>(mean_sum_variance->input(0), subtensor));
    // PortDescriptorUtils::set_port_descriptor_ptr(mean_sum_variance->output(0), std::make_shared<PortDescriptor>(mean_sum_variance->output(0), subtensor));

        // return ov::replace_node_update_name(group_norm_node, reshape_back_node);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_norm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
