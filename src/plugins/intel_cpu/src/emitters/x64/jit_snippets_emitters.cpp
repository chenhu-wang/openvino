// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/port_connector.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op//brgemm_cpu.hpp"
#include "snippets/op/rank_normalization.hpp"

using namespace InferenceEngine;
using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
}

jit_container_emitter::jit_container_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_container_emitter::map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                                                   snippets::lowered::LinearIR::container& expressions) const {
    if (expressions.empty())
        OPENVINO_THROW("Cannot map registers when there is no allocated_emitters provided");
    auto map_regs = [](const std::vector<size_t>& abstract_regs, mapping_info& mapping) {
        auto& abstract_to_physical = mapping.first;
        auto& regs_pool = mapping.second;
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++) {
            const auto abstract = abstract_regs[i];
            auto& physical = physical_regs[i];
            if (abstract_to_physical.count(abstract) == 0) {
                if (regs_pool.empty())
                    OPENVINO_THROW("Cannot map registers for jit_container_emitter: not enough regs in the pool");
                physical = regs_pool.back();
                regs_pool.pop_back();
                abstract_to_physical[abstract] = physical;
            } else {
                physical = abstract_to_physical[abstract];
            }
        }
        return physical_regs;
    };

    for (const auto& expression : expressions) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = expression->get_reg_info();
        std::vector<size_t> in_physical_regs, out_physical_regs;
        switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
            case gpr_to_gpr:
                in_physical_regs = map_regs(in_abstract_regs, gpr_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, gpr_map_pool);
                break;
            case gpr_to_vec:
                // Load Emitters
                in_physical_regs = map_regs(in_abstract_regs, gpr_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, vec_map_pool);
                break;
            case vec_to_gpr:
                // Store Emitters
                in_physical_regs = map_regs(in_abstract_regs, vec_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, gpr_map_pool);
                break;
            case vec_to_vec:
                // Regular operations
                in_physical_regs = map_regs(in_abstract_regs, vec_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, vec_map_pool);
                break;
            default:
                OPENVINO_THROW("Unhandled in_out type");
        }
        expression->set_reg_info({in_physical_regs, out_physical_regs});
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(expression->get_emitter()))
            container->map_abstract_registers(gpr_map_pool,  vec_map_pool, expressions);
    }
}

KernelEmitter::KernelEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_container_emitter(h, isa, expr),
      reg_indexes_idx(abi_param1.getIdx()),
      reg_const_params_idx(abi_param2.getIdx()) {
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    if (!kernel)
        OPENVINO_THROW("KernelEmitter invoked with invalid op argument");
    if (kernel->region.empty())
        OPENVINO_THROW("KernelEmitter invoked with empty body");
    if (kernel->compile_params == nullptr)
        OPENVINO_THROW("KernelEmitter invoked with op::Kernel that contains no compile_params");
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = body.get_master_shape();
    // Note: plugin can prepend master shape with 1 to facilitate parallel execution (usually up to 6D tensor)
    //       so we have to reproduce this behavior here
    master_shape.insert(master_shape.begin(), jcp.parallel_executor_ndims - master_shape.size(), 1);
    const auto& io_exprs = body.get_IO_ops();
    num_inputs = 0;
    num_outputs = 0;
    for (const auto& expr : io_exprs) {
        snippets::lowered::PortDescriptorPtr desc = nullptr;
        element::Type etype;
        switch (expr->get_type()) {
            case snippets::lowered::IOExpression::io_type::INPUT: {
                const auto first_consumer = expr->get_output_port_connector(0)->get_consumers().begin()->get_expr();
                if (ov::is_type<snippets::op::RankNormalization>(first_consumer->get_node())) {
                    desc = first_consumer->get_output_port_descriptor(0);
                } else {
                    desc = expr->get_output_port_descriptor(0);
                }
                etype = expr->get_node()->get_output_element_type(0);
                num_inputs++;
                break;
            }
            case snippets::lowered::IOExpression::io_type::OUTPUT: {
                num_outputs++;
                desc = expr->get_input_port_descriptor(0);
                etype = expr->get_node()->get_input_element_type(0);
                break;
            } default : {
                OPENVINO_THROW("Kernel detected unsupported io_type");
            }
        }
        const auto& shape = desc->get_shape();
        const auto& layout = desc->get_layout();
        OPENVINO_ASSERT(shape.size() == layout.size(), "Shape and layout must have the same length");
        const auto max_dim = *std::max_element(layout.begin(), layout.end());
        OPENVINO_ASSERT(max_dim < shape.size(), "Max layout index can't be larger than the shape size");
        io_shapes.push_back(shape);
        io_data_layouts.push_back(layout);
        io_data_sizes.push_back(etype.size());
    }

    // Initialize pools of gp and vec registers
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    // todo: it's more convenient to use std::set as a pool container (unique and always sorted),
    //  but pools are vectors to align with emit_code signature. Change signature?
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                       [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         reg_indexes_idx, reg_const_params_idx});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);
    snippets::lowered::LinearIR::container mem_access_exprs;
    snippets::lowered::LinearIR::container general_exprs;
    std::set<size_t> unique_buffers;

    for (const auto& expr : body) {
        // Brgemm is a special case since it incorporates input and output (we use onednn kernel)
        // Just like Load & Store it requires offsets calculation
        if (std::dynamic_pointer_cast<snippets::lowered::IOExpression>(expr)) {
            mem_access_exprs.emplace_back(expr);
        } else if (const auto buffer = ov::as_type_ptr<snippets::op::Buffer>(expr->get_node())) {
            const auto buffer_id = buffer->get_id();
            if (unique_buffers.count(buffer_id) == 0) {
                mem_access_exprs.push_back(expr);
                unique_buffers.insert(buffer_id);
            }
        } else {
            general_exprs.emplace_back(expr);
        }
    }
    num_unique_buffers = unique_buffers.size();

    // Note that we can't use reg_indexes_idx or reg_const_params_idx to store data pointers because these two
    // regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    // However we can use reg_indexes_idx and reg_const_params_idx for other operations since we won't need them
    // after offsets calculation
    gpr_map_pool.second.push_back(reg_indexes_idx);
    gpr_map_pool.second.push_back(reg_const_params_idx);
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}

void KernelEmitter::emit_code(const std::vector<size_t> &in,
                              const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    if (m_snippets_segfault_detector)
        build_debug_info();
    emit_impl(in, out);
}

void KernelEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where num_inputs:" << num_inputs << " num_outputs:" << num_outputs << " num_unique_buffers:" << num_unique_buffers
        << " reg_indexes_idx:" << reg_indexes_idx << " reg_const_params_idx:" << reg_const_params_idx << "\n";
}

void KernelEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out) const {
    if (!in.empty())
        OPENVINO_THROW("KernelEmitter got invalid number of inputs. Expected 0, got ", in.size());
    if (!out.empty())
        OPENVINO_THROW("KernelEmitter got invalid number of outputs. Expected 0, got ", out.size());
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    if (data_ptr_regs_idx.size() != num_params)
        OPENVINO_THROW(
            "KernelEmitter: number of inputs and outputs is inconsistent with the number of allocated registers ",
            num_params,
            " data_ptr_regs_idx.size() = ",
            data_ptr_regs_idx.size());
}

void KernelEmitter::init_data_pointers(const Xbyak::Reg64& reg_indexes, const Xbyak::Reg64& reg_const_params,
                                       const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    const auto num_params = num_inputs + num_outputs;
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = master_shape.size() - 1;
    std::vector<std::vector<size_t>> data_offsets(num_params, std::vector<size_t>{});
    auto offset_calculation = [=](const std::vector<size_t>& shape, const std::vector<size_t>& layout, const size_t data_size, bool is_input) {
        // Strides represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    strides: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    strides: s1*s3, s3,       0,  1
        std::vector<size_t> strides(shape.size());
        size_t dim_step = 1;
        strides[shape.size() - 1] = 1;
        for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
            dim_step *= shape[k+1];
            strides[k] = shape[k] != 1 ? dim_step * data_size : 0;
        }
        // Note: this is an extra copy, but let's keep it for clarity
        if (!layout.empty()) {
            std::vector<size_t> reordered_strides(strides.size());
            for (size_t i = 0; i < layout.size(); i++) {
                const auto& src_idx = is_input ? layout[i] : i;
                const auto& dst_idx = is_input ? i : layout[i];
                reordered_strides[dst_idx] = strides[src_idx];
            }
            strides = std::move(reordered_strides);
        }
        // the last stride is ignored, since the entire last dim is processed by kernel
        // and no parallel_for data_ptr offsets can be applied in this case
        strides.pop_back();
        // actual offset size might be larger that the shape size due to 6D scheduling
        strides.insert(strides.begin(), offset_rank - strides.size(), 0);

        return strides;
    };
    for (size_t i = 0; i < num_params; i++) {
        data_offsets[i] = offset_calculation(io_shapes[i],  io_data_layouts[i], io_data_sizes[i], i < num_inputs);
    }
    // master_shape size must be valid in both static and dynamic cases
    std::function<void(Reg64, const std::vector<size_t>&, Reg64)> init_ptr_with_offset;
    init_ptr_with_offset = [&](Reg64 pointer, const std::vector<size_t>& offsets, Reg64 reg_tmp) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_const_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : Reg64(static_cast<int>(*spare_corruptable_gpr));
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_const_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_const_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_const_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterwards
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_const_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_const_params;
        // can corrupt reg_const_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
}
void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    h->preamble();

    Reg64 reg_indexes = Reg64(static_cast<int>(reg_indexes_idx));
    Reg64 reg_const_params = Reg64(static_cast<int>(reg_const_params_idx));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(reg_indexes, reg_const_params, data_ptr_regs);
    for (const auto& expression : body) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = expression->get_reg_info();
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}

LoopBeginEmitter::LoopBeginEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    if (!loop_begin)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid op argument");
    const auto& target_inputs = loop_begin->output(loop_begin->get_output_size() - 1).get_target_inputs();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (target_inputs.size() != 1)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid configuration: the last output must have exactly one "
                       "input attached");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid configuration: the last output must be LoopEnd");
    work_amount = loop_end->get_work_amount();
    evaluate_once = loop_end->get_evaluate_once();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopBeginEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where evaluate_once:" << evaluate_once << " work_amount:" << work_amount << "\n";
}

void LoopBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    if (m_snippets_segfault_detector)
        build_debug_info();
    emit_impl(in, out);
}

void LoopBeginEmitter::validate_arguments(const std::vector<size_t> &in,
                                          const std::vector<size_t> &out) const {
    if (!in.empty())
        OPENVINO_THROW("Invalid inputs size: expected 0 got ", in.size());
    if (out.size() != 1)
        OPENVINO_THROW("Invalid outputs size: expected 1 got ", out.size());
}

void LoopBeginEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
    // todo: In dynamic case we will also need to set broadcasting info here
    Reg64 reg_work_amount = Reg64(static_cast<int>(out.back()));
    Label for_body;
    // save previous register state (if there is an outer loop that uses this reg for example)
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    loop_begin->begin_address = h->getCurr();
}

LoopEndEmitter::LoopEndEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    if (!loop_end)
        OPENVINO_THROW("LoopEndEmitter invoked with invalid op argument");
    loop_begin = loop_end->get_loop_begin();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (!loop_begin)
        OPENVINO_THROW("LoopEndEmitter invoked with invalid configuration: the last arg must be LoopBegin");
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = static_cast<int64_t>(loop_end->get_increment());
    work_amount = static_cast<int64_t>(loop_end->get_work_amount());
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    io_data_size = loop_end->get_element_type_sizes();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopEndEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where num_inputs:" << num_inputs << " num_outputs:" << num_outputs
        << " wa_increment:" << wa_increment << " work_amount:" << work_amount << " evaluate_once:" << evaluate_once << "\n";
}

void LoopEndEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    if (m_snippets_segfault_detector)
        build_debug_info();
    emit_impl(in, out);
}


void LoopEndEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out) const {
    if (out.size() != num_outputs)
        OPENVINO_THROW("Invalid number of out arguments: expected ", num_outputs, " got ", out.size());
    if (in.size() != num_inputs)
        OPENVINO_THROW("Invalid number of in arguments: expected ", num_inputs , " got ", in.size());
    const auto io_size = num_inputs - 1;
    if (ptr_increments.size() != io_size)
        OPENVINO_THROW("Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    if (finalization_offsets.size() != io_size)
        OPENVINO_THROW("Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
}

void LoopEndEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs - 1);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    Reg64 reg_work_amount = Reg64(in.back());
    if (!evaluate_once) {
        for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (ptr_increments[idx] != 0)
                h->add(data_ptr_regs[idx], ptr_increments[idx] * wa_increment * io_data_size[idx]);
        }
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(loop_begin->begin_address);
    }

    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
}

NopEmitter::NopEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

ParameterEmitter::ParameterEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : NopEmitter(h, isa, expr) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

ResultEmitter::ResultEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : NopEmitter(h, isa, expr) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

BroadcastMoveEmitter::BroadcastMoveEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    if (n->get_input_element_type(0) != n->get_output_element_type(0))
        OPENVINO_THROW("BroadcastMoveEmitter supports only equal input and output types but gets: ",
                       n->get_input_element_type(0),
                       " and ",
                       n->get_output_element_type(0));
    byte_size = n->get_input_element_type(0).size();
}

void BroadcastMoveEmitter::emit_impl(const std::vector<size_t>& in,
          const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("BroadcastMove emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void BroadcastMoveEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Xmm xmm_src0 = Xmm(in[0]);
    Vmm vmm_dst  = Vmm(out[0]);

    switch (byte_size) {
        case 4: h->uni_vbroadcastss(vmm_dst, xmm_src0); break;
        case 2: h->vpbroadcastw(vmm_dst, xmm_src0); break;
        case 1: h->vpbroadcastb(vmm_dst, xmm_src0); break;
        default: assert(!"unsupported data type");
    }
}

void BroadcastMoveEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where byte_size:" << byte_size << "\n";
}

ScalarEmitter::ScalarEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    const auto& precision = n->get_output_element_type(0);
    switch (precision) {
        case element::i32: {
            value = ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<int32_t>()[0];
            break;
        }
        case element::f32: {
            value = dnnl::impl::cpu::x64::float2int(ov::as_type_ptr<ov::op::v0::Constant>(n)->cast_vector<float>()[0]);
            break;
        }
        default: {
            OPENVINO_THROW("Scalar emitter doesn't support ", precision);
        }
    }
    push_arg_entry_of("scalar", value, true);
    prepare_table();
}

void ScalarEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("Scalar emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void ScalarEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Vmm vmm_dst  = Vmm(out[0]);
    h->uni_vbroadcastss(vmm_dst, table_val("scalar"));
}

void ScalarEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where value:" << value << "\n";
}

MemoryEmitter::MemoryEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);
}

StoreEmitter::StoreEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("StoreEmitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    store_node = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    if (store_node == nullptr)
        OPENVINO_THROW("Node in expression is not snippets::op::Store in constructor of StoreEmitter!");
    count = store_node->get_count();
    byte_offset = store_node->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("Store emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OPENVINO_THROW("Store CPU emitter isn't initialized for StoreEmitter!");
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreEmitter::emit_data() const {
    store_emitter->emit_data();
}

void StoreEmitter::print_debug_info() const {
    std::cerr << "Node name:" << store_node->get_friendly_name() << std::endl;
    std::cerr << "Emitter type name:" << get_type_name(this) << std::endl;
    if (store_emitter) {
        std::cerr << "underlying jit_store_emitter status is:" << std::endl;
        store_emitter->print_debug_info();
    }
}

LoadEmitter::LoadEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("LoadEmitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    load_node = std::dynamic_pointer_cast<snippets::op::Load>(expr->get_node());
    if (load_node == nullptr)
        OPENVINO_THROW("Node in expression is not snippets::op::Load in constructor of LoadEmitter!");
    count = load_node->get_count();
    byte_offset = load_node->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("Load emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OPENVINO_THROW("Load CPU emitter isn't initialized for LoadEmitter!");
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadEmitter::emit_data() const {
    load_emitter->emit_data();
}

void LoadEmitter::print_debug_info() const {
    std::cerr << "Node name:" << load_node->get_friendly_name() << std::endl;
    std::cerr << "Emitter type name:" << get_type_name(this) << std::endl;
    if (load_emitter) {
        std::cerr << "underlying jit_load_emitter status is:" << std::endl;
        load_emitter->print_debug_info();
    }
}

BroadcastLoadEmitter::BroadcastLoadEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("BroadcastEmitters support only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto broadcast_load = std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(expr->get_node());
    byte_offset = broadcast_load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void BroadcastLoadEmitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("BroadcastLoad emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void BroadcastLoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;
    Reg64 in_reg(in[0]);
    Vmm vmm_dst = Vmm(out[0]);

    // In doesn't really matter if we broadcast or `movss` for vector tails so keep only one version for `BroadcastLoad`,
    // key point here is not to add post-increment, it might be fixed by some other approach in future
    switch (src_prc.size()) {
        case 4: h->uni_vbroadcastss(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        case 2: h->vpbroadcastw(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        case 1: h->vpbroadcastb(vmm_dst, h->ptr[in_reg + byte_offset]); break;
        default: assert(!"unsupported data type");
    }
}

LoadConvertEmitter::LoadConvertEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : MemoryEmitter(h, isa, expr) {
    load_convert_node = ov::as_type_ptr<snippets::op::Load>(expr->get_node());
    if (load_convert_node == nullptr)
        OPENVINO_THROW("Node in expression can not dynamic cast to snippets::op::Load in constructor of LoadConvertEmitter!");
    count = load_convert_node->get_count();
    byte_offset = load_convert_node->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("LoadConvert emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoadConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OPENVINO_THROW("Load CPU emitter isn't initialized for LoadEmitter!");
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void LoadConvertEmitter::emit_data() const {
    load_emitter->emit_data();
}

void LoadConvertEmitter::print_debug_info() const {
    std::cerr << "Node name:" << load_convert_node->get_friendly_name() << std::endl;
    std::cerr << "Emitter type name:" << get_type_name(this) << std::endl;
    if (load_emitter) {
        std::cerr << "underlying jit_load_emitter status is:" << std::endl;
        load_emitter->print_debug_info();
    }
}

StoreConvertEmitter::StoreConvertEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : MemoryEmitter(h, isa, expr) {
    store_convert_node = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    if (store_convert_node == nullptr)
        OPENVINO_THROW("Node in expression can not dynamic cast to snippets::op::Store in constructor of StoreConvertEmitter!");
    count = store_convert_node->get_count();
    byte_offset = store_convert_node->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;

    if (ov::is_type<ov::intel_cpu::StoreConvertTruncation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::truncation));
    } else if (ov::is_type<ov::intel_cpu::StoreConvertSaturation>(expr->get_node())) {
        store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, arithmetic_mode::saturation));
    }
}

void StoreConvertEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("StoreConvert emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void StoreConvertEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OPENVINO_THROW("Store CPU emitter isn't initialized for StoreEmitter!");
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, aux_vec_idxs, aux_gpr_idxs);
}

void StoreConvertEmitter::emit_data() const {
    store_emitter->emit_data();
}

void StoreConvertEmitter::print_debug_info() const {
    std::cerr << "Node name:" << store_convert_node->get_friendly_name() << std::endl;
    std::cerr << "Emitter type name:" << get_type_name(this) << std::endl;
    if (store_emitter) {
        std::cerr << "underlying jit_store_emitter status is:" << std::endl;
        store_emitter->print_debug_info();
    }
}


size_t BrgemmEmitter::get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Input shape is original, so we need to correctly read this data by order
    // Example:
    //      Original shape (shape) = [1, 49, 2, 23]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Transposed shape = [2, 1, 49, 23]
    //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
    OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                    "BrgemmEmitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout[layout.size() - 2];  // `1` in example
    return std::accumulate(shape.cbegin() + idx + 1, shape.end(), 1, std::multiplies<size_t>());
}
size_t BrgemmEmitter::get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Output shape is already transposed, we need to correctly write the data with original shape by the order
    // Example:
    //      Original transposed shape (shape) = [49, 2, 7, 39]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
    //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
    //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
    OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                    "BrgemmEmitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout.size() - 2; // 2 in the example
    const auto dim = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), idx)); // 0 in the example: shape[0] = 49
    return std::accumulate(shape.cbegin() + dim + 1, shape.cend(), 1, std::multiplies<size_t>()); // shape[1] x shape[2] x shape[3] = 2 x 7 x 39
}

BrgemmEmitter::BrgemmEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OPENVINO_ASSERT(!brgemm_node->is_dynamic(), "Snippets don't support code generation for dynamic Brgemm");

    std::vector<size_t> leading_dimensions;
     auto get_layout = [](const std::vector<size_t>& layout, const snippets::VectorDims& io_shape) {
        if (!layout.empty()) return layout;
        std::vector<size_t> default_layout(io_shape.size());
        std::iota(default_layout.begin(), default_layout.end(), 0);
        return default_layout;
    };

    auto init_in_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& input) {
        const auto& layout = get_layout(input->get_layout(), input->get_shape());
        leading_dimensions.push_back(get_in_leading_dim(input->get_shape(), layout));
    };
    auto init_out_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& output) {
        const auto& layout = get_layout(output->get_layout(), output->get_shape());
        leading_dimensions.push_back(get_out_leading_dim(output->get_shape(), layout));
    };

    const auto& input_0_desc = expr->get_input_port_descriptor(0);
    const auto& input_1_desc = expr->get_input_port_descriptor(1);
    const auto& output_desc = expr->get_output_port_descriptor(0);

    init_in_scheduling_params(input_0_desc);
    if (brgemm_node->is_with_data_repacking()) {
        const auto& brgemm_copy = brgemm_node->get_brgemm_copy();
        const auto& allocated_shape = brgemm_copy->get_data_repacking_shape(input_1_desc->get_shape());
        leading_dimensions.push_back(*allocated_shape.rbegin());
    } else {
        init_in_scheduling_params(input_1_desc);
    }
    init_out_scheduling_params(output_desc);

    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    bool brgWithAMX = brgemm_node->is_amx();

    io_data_size = {brg0Prc.size(), brg1Prc.size()};
    if (brgemm_node->get_input_size() == 3)
        io_data_size.push_back(brgemm_node->get_input_element_type(2).size());
    io_data_size.push_back(brgemm_node->get_output_element_type(0).size());

    m_with_comp = brgemm_node->is_with_compensations();
    m_with_scratch = brgemm_node->is_with_scratchpad();

    const auto& output_subtensor = output_desc->get_subtensor();
    const auto& input_0_subtensor = input_0_desc->get_subtensor();
    const auto& input_1_subtensor = input_1_desc->get_subtensor();

    OPENVINO_ASSERT(*(output_subtensor.rbegin() + 1) == *(input_0_subtensor.rbegin() + 1),
                    "Brgemm has different M dimension subtensors on input0 and output");
    OPENVINO_ASSERT(*output_subtensor.rbegin() == *input_1_subtensor.rbegin(),
                    "Brgemm has different N dimension subtensors on input1 and output");
    OPENVINO_ASSERT(*input_0_subtensor.rbegin() == *(input_1_subtensor.rbegin() + 1),
                    "Brgemm has different K dimension subtensors on input0 and input1");

    m_brgCtx.M = *(output_subtensor.rbegin() + 1);
    m_brgCtx.N = *output_subtensor.rbegin();
    m_brgCtx.K = *input_0_subtensor.rbegin();
    m_brgCtx.LDA = leading_dimensions[0];
    m_brgCtx.LDB = leading_dimensions[1];
    m_brgCtx.LDC = leading_dimensions[2];
    m_brgCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
    m_brgCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg1Prc));
    m_brgCtx.beta = brgemm_node->get_beta();

    initBrgemm(m_brgCtx, m_brgKernel, brgWithAMX);

    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
    if (m_with_scratch)
        m_load_offset_scratch = brgemm_node->get_offset_scratch();
}

std::set<std::vector<element::Type>> BrgemmEmitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "BrgemmEmitter::get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmCPU::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmCPU::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmCPU::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmCPU::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OPENVINO_THROW("BrgemmEmitter got BrgemmCPU node with unsupported type");
    }
}

void BrgemmEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (m_with_scratch && in.size() != 3) {
        IE_THROW() << "BRGEMM Emitter expects 3 inputs if there are compensations/wsp";
    }
}

void BrgemmEmitter::initBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) {
    brgemm_t brgDesc;
    const bool is_int8 = utils::one_of(ctx.dt_in0, data_type::u8, data_type::s8) && utils::one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx ? isa_undef
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : (is_int8 ? avx512_core_vnni : avx512_core);
    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
                                   false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, nullptr);
    if (status != dnnl_success)
        OPENVINO_THROW("BrgemmEmitter cannot initialize brgemm descriptor due to invalid params");

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx)
        amx_tile_configure(ctx.palette);

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success)
        OPENVINO_THROW("BrgemmEmitter cannot create brgemm kernel due to invalid params");
    brgKernel.reset(brgKernel_);
}

void BrgemmEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 input_0(static_cast<int>(in[0]));
        Xbyak::Reg64 input_1(static_cast<int>(in[1]));
        Xbyak::Reg64 input_2(static_cast<int>(m_with_scratch ? in[2] : 0));  // scratch. Default reg index is 0 if there isn't scratch
        Xbyak::Reg64 output_0(static_cast<int>(out[0]));
        emit_brgemm_kernel_call(m_brgKernel.get(),
                                m_brgCtx,
                                input_0,
                                input_1,
                                input_2,
                                output_0,
                                m_load_offset_a,
                                m_load_offset_b,
                                m_load_offset_scratch,
                                m_store_offset_c);
    } else {
        OPENVINO_THROW("BrgemmEmitter requires at least avx512_core instruction set");
    }
}

void BrgemmEmitter::emit_brgemm_kernel_call(const brgemm_kernel_t *brg_kernel, const brgemmCtx& ctx,
                                            Reg64 addr_A, Reg64 addr_B, Reg64 scratch, Reg64 addr_C,
                                            const size_t in0_kernel_offset, const size_t in1_kernel_offset,
                                            const size_t in2_kernel_offset, const size_t out0_kernel_offset) const {
    if (ctx.is_with_amx) {
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                         h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // save function address in gpr to pass in call instruction
        const auto& overload = static_cast<status_t(*)(const char*)>(amx_tile_configure);
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(overload));
        h->mov(abi_param1, reinterpret_cast<uintptr_t>(ctx.palette));

        // align stack on 16-byte as ABI requires
        // note that RBX must not be changed by the callee
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        h->call(h->rbp);

        h->add(h->rsp, h->rbx);
        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);
    }

    internal_call_preamble();

    // save function address in gpr to pass in call instruction
    const auto& brgemm_kernel_overload = static_cast<void (*)(const brgemm_kernel_t*,
                                                              const void*,
                                                              const void*,
                                                              void*,
                                                              void*,
                                                              int)>(kernel_execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(brgemm_kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), addr_A);
    h->uni_vmovq(Xmm(1), addr_B);
    h->uni_vmovq(Xmm(2), addr_C);
    if (m_with_scratch)
        h->uni_vmovq(Xmm(3), scratch);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(brg_kernel));
    data_ptr_reg(Xmm(0), abi_param2, in0_kernel_offset);
    data_ptr_reg(Xmm(1), abi_param3, in1_kernel_offset);
    data_ptr_reg(Xmm(2), abi_param4, out0_kernel_offset);

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 6;  // count of function brgemm_kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);
    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);

    // Push the remaining parameters on the stack
    if (m_with_scratch) {
        h->uni_vmovq(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], Xmm(3));
        if (in2_kernel_offset) h->add(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], in2_kernel_offset);
    } else {
        h->mov(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_not_param1, static_cast<int>(m_with_comp));
    h->mov(h->qword[h->rsp + (abi_param_count + 1) * gpr_size], abi_not_param1);
#else
    if (m_with_scratch) {
        data_ptr_reg(Xmm(3), abi_param5, in2_kernel_offset);
    } else {
        h->mov(abi_param5, reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_param6, static_cast<int>(m_with_comp));
#endif

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

#ifdef _WIN32
    h->add(h->rsp, num_args_passed_on_stack * gpr_size);
#endif

    internal_call_postamble();
}

void BrgemmEmitter::kernel_execute(const brgemm_kernel_t *brg_kernel,
                                   const void *A, const void *B, void *C, void *scratch, int with_comp) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = A;
    brgemm_p.ptr_B = B;
    brgemm_p.ptr_C = C;
    brgemm_p.ptr_D = C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(with_comp);
    brgemm_p.do_apply_comp = static_cast<size_t>(with_comp);
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

void BrgemmEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where m_M:" << m_M << " m_K:" << m_K << " m_K_blk:" << m_K_blk << " m_K_tail:" << m_K_tail
        << " m_N:" << m_N << " m_N_blk:" << m_N_blk << " m_N_tail:" << m_N_tail
        << " m_brg0VnniFactor:" << m_brg0VnniFactor << " m_N_blk_loop:" << m_N_blk_loop << " m_K_blk_loop:" << m_K_blk_loop
        << " m_load_offset_a:" << m_load_offset_a << " m_load_offset_b:" << m_load_offset_b << " m_load_offset_scratch:" << m_load_offset_scratch
        << " m_store_offset_c:" << m_store_offset_c
        << " m_with_scratch:" << m_with_scratch << " m_with_comp:" << m_with_comp << "\n";
}

BrgemmCopyBEmitter::BrgemmCopyBEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node());
    if (!brgemm_repack)
        OPENVINO_THROW("BrgemmCopyBEmitters expects BrgemmCopyB node");

    m_brgemm_prc_in0 = brgemm_repack->get_src_element_type();
    m_brgemm_prc_in1 = brgemm_repack->get_input_element_type(0);
    m_brgemmVNNIFactor = 4 / m_brgemm_prc_in0.size();
    m_with_comp = brgemm_repack->is_with_compensations();
    m_in_offset = brgemm_repack->get_offset_in();
    m_out_offset = brgemm_repack->get_offset_out();
    if (m_with_comp)
        m_comp_offset = brgemm_repack->get_offset_compensations();

    const auto& in_desc = expr->get_input_port_descriptor(0);
    const auto& layout = in_desc->get_layout();
    const auto& original_shape = in_desc->get_shape();
    auto transposed_shape = original_shape;
    size_t leading_dimension = *(original_shape.rbegin());
    if (!layout.empty()) {
        transposed_shape = snippets::utils::get_planar_vdims(original_shape, layout);
        leading_dimension = BrgemmEmitter::get_in_leading_dim(original_shape, layout);
    }

    m_N = *(transposed_shape.rbegin());
    m_K = *(transposed_shape.rbegin() + 1);

    m_N_blk = brgemm_repack->get_n_block_size();
    m_K_blk = brgemm_repack->get_k_block_size();

    m_N_tail = m_N % m_N_blk;
    m_K_tail = m_K % m_K_blk;
    m_LDB = m_brgemm_prc_in1 == ov::element::f32 ? leading_dimension : rnd_up(m_N, m_N_blk);

    const auto dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(m_brgemm_prc_in0));
    const auto dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(m_brgemm_prc_in1));

    const bool isAMXSupported = mayiuse(avx512_core_amx);
    const auto use_amx = isAMXSupported && m_brgemm_prc_in0 != ov::element::f32 && (m_K % m_brgemmVNNIFactor == 0) && (m_N % m_brgemmVNNIFactor == 0);
    init_brgemm_copy(m_kernel, leading_dimension, m_N_blk, m_N_tail, m_LDB, m_K - m_K_tail, use_amx, dt_in0, dt_in1);
}

void BrgemmCopyBEmitter::init_brgemm_copy(std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                                          size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
                                          bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(N_blk);
    brgCopyKernelConf.wei_tag = dnnl_abcd;  // What's about other ranks?
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = static_cast<dim_t>(LDB);
    brgCopyKernelConf.N =  static_cast<dim_t>(N);
    brgCopyKernelConf.N_tail =  static_cast<dim_t>(N_tail);
    brgCopyKernelConf.N_blk =  static_cast<dim_t>(N_blk);
    brgCopyKernelConf.K =  static_cast<dim_t>(K);
    brgCopyKernelConf.K_blk =  static_cast<dim_t>(K);
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    if (is_with_amx) {
        brgCopyKernelConf.isa = avx512_core_amx;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    auto status = matmul::create_brgemm_matmul_copy_b(kernel, &brgCopyKernelConf);
    if (status != dnnl_success)
        OPENVINO_THROW("BrgemmRepackEmitter cannot create kernel due to invalid params");
}

void BrgemmCopyBEmitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out) const {
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 src(static_cast<int>(in[0]));
        Xbyak::Reg64 dst(static_cast<int>(out[0]));
        Xbyak::Reg64 comp(static_cast<int>(0));  // Compensations. Default reg idx is 0 if there aren't the compensations
        if (m_with_comp) {
            if (out.size() != 2) {
                OPENVINO_THROW("BrgemmCopyBEmitter with compensations requires separate register for them");
            }
            comp = Xbyak::Reg64(static_cast<int>(out[1]));
        }

        const size_t data_size = m_brgemm_prc_in1.size();
        for (size_t nb = 0; nb < div_up(m_N, m_N_blk); nb++) {
            const size_t offset_in = m_in_offset + nb * m_N_blk * data_size;
            const size_t offset_out = m_out_offset + nb * m_N_blk * m_brgemmVNNIFactor * data_size;
            const size_t offset_comp = m_with_comp ? m_comp_offset + nb * m_N_blk * sizeof(int32_t) : 0;

            const bool is_N_tail = (m_N - nb * m_N_blk < m_N_blk);
            const auto current_N_blk = is_N_tail ? m_N_tail : m_N_blk;

            emit_kernel_call(m_kernel.get(), src, dst, comp, current_N_blk, m_K, offset_in, offset_out, offset_comp);
        }
    } else {
        OPENVINO_THROW("BrgemmCopyBEmitter requires at least avx512_core instruction set");
    }
}

void BrgemmCopyBEmitter::emit_kernel_call(const matmul::jit_brgemm_matmul_copy_b_t* kernel, Reg64 src, Reg64 dst, Reg64 comp,
                                          size_t N, size_t K, size_t offset_in, size_t offset_out, size_t offset_comp) const {
    internal_call_preamble();

    const auto data_ptr = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
#ifdef _WIN32
    const auto push_value = [&](size_t value, size_t index) {
        // Firstly we need to move integer to GPR. Then we can move value from GPR to stack
        h->mov(abi_not_param1, value);
        h->mov(h->qword[h->rsp + index * gpr_size], abi_not_param1);
    };
#endif

    // save function address in gpr to pass in call instruction
    const auto &kernel_overload = static_cast<void (*)(matmul::jit_brgemm_matmul_copy_b_t*,
                                                       const void*,
                                                       const void*,
                                                       const void*,
                                                       size_t,
                                                       size_t)>(execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), src);
    h->uni_vmovq(Xmm(1), dst);
    if (m_with_comp)
        h->uni_vmovq(Xmm(2), comp);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(kernel));

    data_ptr(Xmm(0), abi_param2, offset_in);
    data_ptr(Xmm(1), abi_param3, offset_out);
    if (m_with_comp) {
        data_ptr(Xmm(2), abi_param4, offset_comp);
    } else {
        h->mov(abi_param4, reinterpret_cast<uintptr_t>(nullptr));
    }

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 6;  // count of function kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);

    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);
    push_value(N, abi_param_count + 0);
    push_value(K, abi_param_count + 1);
#else
    h->mov(abi_param5, N);
    h->mov(abi_param6, K);
#endif

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

#ifdef _WIN32
        h->add(h->rsp, gpr_size * num_args_passed_on_stack);
#endif

    internal_call_postamble();
}

void BrgemmCopyBEmitter::execute(matmul::jit_brgemm_matmul_copy_b_t *kernel, const void *src,
                                 const void *dst, const void *comp, size_t N, size_t K) {
    if (!kernel)
        OPENVINO_THROW("Kernel for `brgemm_copy_b` hasn't been created");

    auto ctx = dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = N;
    ctx.src = src;
    ctx.tr_src = dst;
    ctx.compensation_ptr = comp;
    ctx.zp_a_compensation_ptr = nullptr;
    ctx.zp_a_neg_value_ptr = nullptr;
    ctx.current_K_start = 0;
    ctx.current_K_iters = K;

    (*kernel)(&ctx);
}

void BrgemmCopyBEmitter::print_debug_info() const {
    std::cerr << "Emitter type name:" << get_type_name(this) << "\n";
    std::cerr << "where m_LDB:" << m_LDB << " m_K:" << m_K << " m_K_blk:" << m_K_blk << " m_K_tail:" << m_K_tail
        << " m_N:" << m_N << " m_N_blk:" << m_N_blk << " m_N_tail:" << m_N_tail
        << " m_brgemm_prc_in0:" << m_brgemm_prc_in0 << " m_brgemm_prc_in1:" << m_brgemm_prc_in1
        << " m_brgemmVNNIFactor:" << m_brgemmVNNIFactor << " m_with_comp:" << m_with_comp
        << " m_in_offset:" << m_in_offset << " m_out_offset:" << m_out_offset << " m_comp_offset:" << m_comp_offset << "\n";
}

HorizonEmitter::HorizonEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    if (ov::is_type<const snippets::op::HorizonMax>(expr->get_node())) {
        m_op_type = OpType::max;
    } else if (ov::is_type<const snippets::op::HorizonSum>(expr->get_node())) {
        m_op_type = OpType::sum;
    } else {
        OPENVINO_THROW("HorizonEmitter exprects HorizonMax or HorizonSum ops");
    }
}

void HorizonEmitter::emit_impl(const std::vector<size_t>& in,
                                    const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("HorizonMax emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void HorizonEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Vmm src_vmm = Vmm(in[0]);
    Vmm dst_vmm = Vmm(out[0]);
    Vmm aux_vmm = Vmm(aux_vec_idxs[0]);

    if (in[0] != out[0])
        h->uni_vmovups(dst_vmm, src_vmm);
    if (isa == dnnl::impl::cpu::x64::avx512_core) {
        Zmm dst_zmm = Zmm(out[0]);
        Zmm aux_zmm = Zmm(aux_vec_idxs[0]);
        h->vshuff32x4(aux_zmm, dst_zmm, dst_zmm, 0x4E);
        perform_op<Zmm>(dst_zmm, dst_zmm, aux_zmm);
        h->vshuff32x4(aux_zmm, dst_zmm, dst_zmm, 0xB1);
        perform_op<Zmm>(dst_zmm, dst_zmm, aux_zmm);
    } else if (isa == dnnl::impl::cpu::x64::avx2) {
        Ymm dst_ymm = Ymm(out[0]);
        Ymm aux_ymm = Ymm(aux_vec_idxs[0]);
        h->vperm2i128(aux_ymm, dst_ymm, dst_ymm, 0x01);
        perform_op<Ymm>(dst_ymm, dst_ymm, aux_ymm);
    }
    h->uni_vshufps(aux_vmm, dst_vmm, dst_vmm, 0x4E);
    perform_op<Xmm>(dst_vmm, dst_vmm, aux_vmm);
    h->uni_vshufps(aux_vmm, dst_vmm, dst_vmm, 0xB1);
    perform_op<Xmm>(dst_vmm, dst_vmm, aux_vmm);
}

template<typename Vmm>
void HorizonEmitter::perform_op(const Vmm &vmm1, const Vmm &vmm2, const Vmm &vmm3) const {
    switch (m_op_type) {
        case OpType::max:
            h->uni_vmaxps(vmm1, vmm2, vmm3);
            break;
        case OpType::sum:
            h->uni_vaddps(vmm1, vmm2, vmm3);
            break;
        default:
            assert(!"Unsupported horizontal operation.");
    }
}

FillEmitter::FillEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa, ov::element::f32, emitter_in_out_map::vec_to_vec) {
    const auto fill = ov::as_type_ptr<snippets::op::Fill>(expr->get_node());
    if (fill->get_element_type().size() != 4) {
        OPENVINO_THROW("Fill emitter supports only 4 Byte element types but gets: ", fill->get_element_type());
    }

    offset = fill->get_offset();
    fill_value = fill->get_fill_value();
    if (!is_optimized())
        push_arg_entry_of("value", fill_value, true);
    prepare_table();
}

size_t FillEmitter::aux_gprs_count() const {
    // Optimized version (fill full vector by zero) doesn't need additional register
    if (is_optimized())
        return 0;
    // + 1 reg for table value in full vector case
    if (is_full_reg())
        return 1;
    // + 1 reg for temp reg for mask in avx512
    return one_of(host_isa_, dnnl::impl::cpu::x64::avx512_core) ? 2 : 1;
}

void FillEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        OPENVINO_THROW("Fill emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void FillEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Vmm src_vmm = Vmm(in[0]);
    Vmm dst_vmm = Vmm(out[0]);

    if (is_full_reg())
        fill_full<Vmm>(dst_vmm);
    else
        fill_tail<Vmm>(src_vmm, dst_vmm);
}

template <typename Vmm>
void FillEmitter::fill_full(const Vmm& dst_vmm) const {
    // Optimized impl for zero
    if (is_optimized()) {
        h->uni_vpxor(dst_vmm, dst_vmm, dst_vmm);
        return;
    }

    h->uni_vbroadcastss(dst_vmm, table_val("value"));
}

template <typename Vmm>
void FillEmitter::fill_tail(const Vmm& src_vmm, const Vmm& dst_vmm) const {
    if (one_of(host_isa_, dnnl::impl::cpu::x64::avx512_core)) {
        uint64_t tail_mask = 1;
        tail_mask = ~((tail_mask << offset) - tail_mask);
        h->mov(Reg64(aux_gpr_idxs[0]), tail_mask);
        h->kmovq(k_mask, Reg64(aux_gpr_idxs[0]));
        h->vblendmps(dst_vmm | k_mask, src_vmm, table_val("value"));
    } else if (one_of(host_isa_, dnnl::impl::cpu::x64::avx2, dnnl::impl::cpu::x64::sse41)) {
        uint8 imm = 1;
        imm = ~((imm << offset) - imm);  // shift load_num bit
        if (host_isa_ == dnnl::impl::cpu::x64::sse41 && src_vmm.getIdx() != dst_vmm.getIdx()) {
            h->uni_vmovups(dst_vmm, src_vmm);
            h->uni_vblendps(dst_vmm, dst_vmm, table_val("value"), imm);
        } else {
            h->uni_vblendps(dst_vmm, src_vmm, table_val("value"), imm);
        }
    }
}

}  // namespace intel_cpu
}  // namespace ov
