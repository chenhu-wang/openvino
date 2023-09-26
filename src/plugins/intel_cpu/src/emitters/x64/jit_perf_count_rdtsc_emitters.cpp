// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"
#include "jit_perf_count_rdtsc_emitters.hpp"
#include <cpu/x64/jit_generator.hpp>

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
using namespace Xbyak::util;

namespace ov {
namespace intel_cpu {

jit_perf_count_rdtsc_start_emitter::jit_perf_count_rdtsc_start_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                            const std::shared_ptr<ov::Node>& n) : jit_emitter(host, host_isa) {
    auto start_op = ov::as_type_ptr<ov::intel_cpu::PerfCountRdtscBegin>(n);
    m_current_count = &(start_op->start_count);
}

size_t jit_perf_count_rdtsc_start_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_rdtsc_start_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    h->push(h->rax);
    h->push(h->rdx);

    // The EDX register is loaded with the high-order 32 bits of the MSR and the EAX register is loaded with the low-order 32 bits.
    h->lfence();
    h->rdtsc();
    h->lfence();
    h->shl(h->rdx, 0x20);     // shift to higher half of rdx 0x20(32)
    h->or_(h->rdx, h->rax);   // rdx has current tsc

    h->mov(h->rax, reinterpret_cast<size_t>(m_current_count));
    h->mov(qword[h->rax], h->rdx);

    h->pop(h->rdx);
    h->pop(h->rax);
}

///////////////////jit_perf_count_rdtsc_end_emitter////////////////////////////////////
jit_perf_count_rdtsc_end_emitter::jit_perf_count_rdtsc_end_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
    const std::shared_ptr<ov::Node>& n) : jit_emitter(host, host_isa) {
        auto end_op = ov::as_type_ptr<ov::intel_cpu::PerfCountRdtscEnd>(n);
        m_accumulation = &(end_op->accumulation);
        m_iteration = &(end_op->iteration);
        m_start_count = &(end_op->get_pc_begin()->start_count);
}

size_t jit_perf_count_rdtsc_end_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_rdtsc_end_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    h->push(h->rax);
    h->push(h->rdx);

    h->lfence();
    h->rdtsc();
    h->lfence();
    h->shl(h->rdx, 0x20);
    h->or_(h->rdx, h->rax);  // rdx has current tsc

    // tsc duration
    h->mov(h->rax, reinterpret_cast<size_t>(m_start_count));
    h->sub(h->rdx, qword[h->rax]);  // rdx has tsc duration

    // m_accumulation = m_accumulation + tsc duration
    h->mov(h->rax, reinterpret_cast<size_t>(m_accumulation));
    h->add(h->rdx, qword[h->rax]);
    h->mov(qword[h->rax], h->rdx);

    // iteration++
    h->mov(h->rax, reinterpret_cast<size_t>(m_iteration));
    h->mov(h->rdx, qword[h->rax]);
    h->add(h->rdx, 0x01);
    h->mov(qword[h->rax], h->rdx);

    h->pop(h->rdx);
    h->pop(h->rax);
}

}   // namespace intel_cpu
}   // namespace ov
