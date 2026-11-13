/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "src/ir/transforms/lower_l2/ccu_collectives.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/ccu_collective_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto::ir {
bool IsCcuAllGather(const CallPtr& call) {
  return IsOp(call, "pld.tensor.allgather") && call->GetKwarg<std::string>("engine", "default") == "ccu";
}
bool IsCcuCollective(const CallPtr& call) { return IsCcuAllGather(call); }

namespace {

// SimplerCcuRegistrationBinding::channels has SIMPLER_CCU_MAX_RANKS entries.
constexpr int64_t kMaxCcuRanks = 64;

bool IsOrchAt(const FunctionPtr& func, Level level) {
  return (IsOrchestrationLike(func->func_type_) || func->role_ == Role::Orchestrator) &&
         func->level_ == level;
}

ExprPtr Index(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

TypePtr OrderType(const Span& span) {
  return std::make_shared<TensorType>(std::vector<ExprPtr>{Index(1, span)}, DataType::UINT8);
}

class Names : public IRVisitor {
 public:
  std::set<std::string> used;
  std::string Fresh(const std::string& prefix) {
    while (true) {
      auto name = prefix + std::to_string(next_++);
      if (used.insert(name).second) return name;
    }
  }

 protected:
  void VisitVarLike_(const VarPtr& var) override {
    used.insert(var->name_hint_);
    used.insert(auto_name::GetCompatibleBaseName(var->name_hint_));
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "pld.tensor.alloc_window_buffer")) used.insert(call->GetKwarg<std::string>("name"));
    IRVisitor::VisitExpr_(call);
  }

 private:
  size_t next_ = 0;
};

// Discovery also rejects control flow before rewriting it. A branch may make
// participants execute different CCU sequences; repeated device execution is
// not qualified yet. Host rank-dispatch loops are handled separately below.
class CcuFinder : public IRVisitor {
 public:
  explicit CcuFinder(FunctionPtr func) : func_(std::move(func)) {}
  bool found = false;
  std::set<int64_t> rank_counts;

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (IsCcuCollective(call)) {
      CHECK_SPAN(IsOrchAt(func_, Level::CHIP), call->span_)
          << "CCU collective is supported only in CHIP orchestration, not HOST/InCore";
      CHECK_SPAN(depth_ == 0, call->span_) << "CHIP CCU collective requires a straight-line sequence; loops "
                                              "and conditional calls are unsupported";
      found = true;
      auto target = As<DistributedTensorType>(call->args_[1]->GetType());
      if (auto ranks = As<ConstInt>(target->shape_[0])) rank_counts.insert(ranks->value_);
    }
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    CHECK_SPAN(
        !IsOp(submit, "pld.tensor.allgather") || submit->GetKwarg<std::string>("engine", "default") != "ccu",
        submit->span_)
        << "CCU collective direct-op Submit is unsupported; submit a CHIP callable instead";
    IRVisitor::VisitExpr_(submit);
  }
  void VisitStmt_(const ForStmtPtr& op) override { Nested(op); }
  void VisitStmt_(const WhileStmtPtr& op) override { Nested(op); }
  void VisitStmt_(const IfStmtPtr& op) override { Nested(op); }

 private:
  template <typename T>
  void Nested(const T& op) {
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }
  FunctionPtr func_;
  int depth_ = 0;
};

// CCU allgather requires static positive packed ND views with no partial
// valid region. Dtype and rank restrictions are checked by its lowering.
std::string ValidateView(const DistributedTensorTypePtr& type, const Span& span) {
  CHECK_SPAN(type && !type->shape_.empty(), span) << "CCU collective requires a nonempty static shape";
  std::string key;
  int64_t stride = 1;
  const auto& view = type->tensor_view_;
  CHECK_SPAN(!view || view->layout == TensorLayout::ND, span)
      << "CCU collective requires contiguous ND views";
  CHECK_SPAN(!view || view->stride.empty() || view->stride.size() == type->shape_.size(), span)
      << "CCU collective requires full contiguous strides";
  CHECK_SPAN(!view || view->valid_shape.empty() || view->valid_shape.size() == type->shape_.size(), span)
      << "CCU collective requires a full valid view";
  for (size_t i = type->shape_.size(); i-- > 0;) {
    auto extent = As<ConstInt>(type->shape_[i]);
    CHECK_SPAN(extent && extent->value_ > 0, span) << "CCU collective requires static positive dimensions";
    if (view && !view->stride.empty()) {
      auto actual = As<ConstInt>(view->stride[i]);
      CHECK_SPAN(actual && actual->value_ == stride, span) << "CCU collective requires contiguous ND strides";
    }
    if (view && !view->valid_shape.empty()) {
      auto valid = As<ConstInt>(view->valid_shape[i]);
      CHECK_SPAN(valid && valid->value_ == extent->value_, span)
          << "CCU collective requires a full valid view";
    }
    CHECK_SPAN(extent->value_ <= std::numeric_limits<int64_t>::max() / stride, span)
        << "CCU collective element count exceeds INT64_MAX";
    stride *= extent->value_;
  }
  for (const auto& dim : type->shape_) key += "_" + std::to_string(As<ConstInt>(dim)->value_);
  return key;
}

using Chips = std::set<std::string>;

class CcuLowerer : public IRMutator {
 public:
  CcuLowerer(Names* names, std::map<std::string, FunctionPtr>* kernels) : names_(names), kernels_(kernels) {}

  StmtPtr Lower(const StmtPtr& body) {
    auto lowered = VisitStmt(body);
    allocations_.push_back(lowered);
    return SeqStmts::Flatten(std::move(allocations_), body->span_);
  }

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto call = As<Call>(IRMutator::VisitExpr_(op));
    return IsCcuAllGather(call) ? LowerAllGather(call) : call;
  }

 private:
  VarPtr Order(const std::string& key, const Span& span) {
    auto [token, inserted] = tokens_.try_emplace(key);
    if (inserted) {
      auto shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{Index(1, span)}, span);
      auto alloc = OpRegistry::GetInstance().Create(
          "tensor.create", {shape}, {{"dtype", DataType::UINT8}, {"layout", TensorLayout::ND}}, span);
      token->second = std::make_shared<Var>(names_->Fresh("_ccu_order_"), alloc->GetType(), span);
      allocations_.push_back(std::make_shared<AssignStmt>(token->second, alloc, span));
    }
    return token->second;
  }

  ExprPtr LowerAllGather(const CallPtr& call) {
    const auto& span = call->span_;
    auto input = As<DistributedTensorType>(call->args_[0]->GetType());
    auto target = As<DistributedTensorType>(call->args_[1]->GetType());
    auto signal = As<DistributedTensorType>(call->args_[2]->GetType());
    CHECK_SPAN(input, span) << "CCU allgather input must be a DistributedTensor window";
    ValidateView(input, span);
    auto shape_key = ValidateView(target, span);
    ValidateView(signal, span);
    CHECK_SPAN(input->dtype_ == DataType::BF16, span) << "CCU allgather supports only BF16";
    const int64_t nranks = As<ConstInt>(target->shape_[0])->value_;
    CHECK_SPAN(nranks >= 2 && nranks <= kMaxCcuRanks, span)
        << "CCU allgather requires 2..64 ranks, matching the simpler registration ABI";
    CHECK_SPAN(signal->shape_.size() == 2 && As<ConstInt>(signal->shape_[0])->value_ == nranks &&
                   As<ConstInt>(signal->shape_[1])->value_ == 1,
               span)
        << "CCU allgather requires signal shape [NR, 1] matching target [NR, N]";
    CHECK_SPAN(As<ConstInt>(input->shape_[1])->value_ <= std::numeric_limits<int64_t>::max() / (2 * nranks),
               span)
        << "CCU allgather target byte size exceeds INT64_MAX";
    const auto* handler = PassContext::Current()->GetBackendHandler();
    CHECK_SPAN(handler && handler->SupportCcu(), span)
        << "CCU allgather is unsupported by this backend; compiler lowering requires Ascend950";
    const std::string entry = "dtype_cpp=bfloat16_t,nranks_cpp=" + std::to_string(nranks);
    auto token = Order(std::string(kCcuAllGatherTemplate) + ":" + entry, span);
    auto name = "__builtin_allgather_ccu__bf16_p" + std::to_string(nranks) + shape_key;
    if (!kernels_->count(name)) {
      // Only target carries a ctx: simpler requires the selected CCU domain
      // exactly once in task scalar arguments. All operands belong to it.
      std::vector<VarPtr> params{
          std::make_shared<Var>("input", std::make_shared<TensorType>(input->shape_, input->dtype_), span),
          std::make_shared<Var>(
              "target", std::make_shared<DistributedTensorType>(target->shape_, target->dtype_), span),
          std::make_shared<Var>("signal", std::make_shared<TensorType>(signal->shape_, signal->dtype_), span),
          std::make_shared<Var>("order", OrderType(span), span)};
      auto body = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{params[1]}, span);
      std::vector<std::pair<std::string, std::any>> attrs{
          {kAttrBuiltinTemplateDir, std::string(kCcuAllGatherTemplate)},
          {kAttrBuiltinTemplateVars, entry},
          {kAttrRequireContiguousOperands, true}};
      (*kernels_)[name] = std::make_shared<Function>(
          name, params,
          std::vector<ParamDirection>{ParamDirection::In, ParamDirection::InOut, ParamDirection::InOut,
                                      ParamDirection::InOut},
          std::vector<TypePtr>{params[1]->GetType()}, body, span, FunctionType::AIV, std::nullopt,
          std::nullopt, attrs);
    }
    return std::make_shared<Call>(
        std::make_shared<GlobalVar>(name),
        std::vector<ExprPtr>{call->args_[0], call->args_[1], call->args_[2], token},
        std::vector<std::pair<std::string, std::any>>{},
        std::vector<std::pair<std::string, std::any>>{{kAttrCoreNum, Index(1, span)}}, call->GetType(), span);
  }

  // Each Lowerer instance owns one CHIP invocation's tokens. A shared InOut
  // tensor creates a WAW chain only for repeated uses of the same CCU entry.
  // No remote access, Host parameter, communication window or ctx is needed.
  Names* names_;
  std::map<std::string, VarPtr> tokens_;
  std::vector<StmtPtr> allocations_;
  std::map<std::string, FunctionPtr>* kernels_;
};

// A deliberately local binding analysis: direct window/SSA aliases, constant
// devices or one bounded static rank-dispatch loop. No cross-function provenance
// or general domain proof is introduced. Coverage is collected for ALL dispatch
// consumers so another call cannot silently enlarge a CCU window's domain.
class HostBindings : public IRVisitor {
 public:
  explicit HostBindings(const Chips& chips) : chips_(chips) {}
  std::map<std::string, std::set<int64_t>> devices;
  std::map<std::string, std::set<const Var*>> buffers;
  std::unordered_map<const Var*, std::set<int64_t>> coverage;

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (auto var = AsVarLike(op->var_)) {
      if (auto ci = As<ConstInt>(op->value_)) constants_[var.get()] = ci->value_;
      if (auto alias = AsVarLike(op->value_)) {
        auto it = windows_.find(alias.get());
        if (it != windows_.end()) windows_[var.get()] = it->second;
      }
      if (auto call = As<Call>(op->value_); IsOp(call, "pld.tensor.window")) {
        if (auto buffer = AsVarLike(call->args_[0])) windows_[var.get()] = buffer.get();
      }
    }
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    ++loop_depth_;
    auto start = Constant(op->start_);
    auto stop = Constant(op->stop_);
    auto step = Constant(op->step_);
    if (start && stop && step && *start >= 0 && *step == 1 && *stop > *start &&
        *stop - *start <= kMaxCcuRanks) {
      auto& devices = loop_devices_[op->loop_var_.get()];
      for (int64_t rank = *start; rank < *stop; ++rank) devices.insert(rank);
    }
    IRVisitor::VisitStmt_(op);
    loop_devices_.erase(op->loop_var_.get());
    --loop_depth_;
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    ++unsupported_depth_;
    IRVisitor::VisitStmt_(op);
    --unsupported_depth_;
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    ++unsupported_depth_;
    IRVisitor::VisitStmt_(op);
    --unsupported_depth_;
  }
  void VisitExpr_(const CallPtr& call) override {
    Analyze(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    Analyze(call);
    IRVisitor::VisitExpr_(call);
  }

 private:
  std::optional<int64_t> Constant(const ExprPtr& expr) const {
    if (auto ci = As<ConstInt>(expr)) return ci->value_;
    if (auto var = AsVarLike(expr)) {
      auto it = constants_.find(var.get());
      if (it != constants_.end()) return it->second;
    }
    return std::nullopt;
  }
  template <typename T>
  void Analyze(const T& call) {
    const bool ccu = chips_.count(call->op_->name_);
    auto device = call->template GetAttr<ExprPtr>(kAttrDevice);
    if (!ccu && !device) return;
    std::set<int64_t> ranks;
    bool rank_loop = false;
    if (auto ci = Constant(device); ci && *ci >= 0) {
      ranks.insert(*ci);
    }
    if (auto var = AsVarLike(device)) {
      auto it = loop_devices_.find(var.get());
      if (it != loop_devices_.end()) {
        ranks = it->second;
        rank_loop = true;
      }
    }
    if (ccu) {
      CHECK_SPAN(!ranks.empty() && unsupported_depth_ == 0 && loop_depth_ == (rank_loop ? 1 : 0), call->span_)
          << "CCU CHIP dispatch requires constant devices or one static rank loop of at most 64 devices; "
             "conditional/repeated dispatch and dynamic device bindings are unsupported";
      devices[call->op_->name_].insert(ranks.begin(), ranks.end());
    }
    for (const auto& arg : call->args_) {
      if (!As<DistributedTensorType>(arg->GetType())) continue;
      auto var = AsVarLike(arg);
      auto it = windows_.find(var.get());
      if (ccu) {
        CHECK_SPAN(it != windows_.end(), call->span_)
            << "CCU CHIP dispatch requires direct Host window bindings or SSA aliases";
      }
      if (it == windows_.end()) continue;
      // Unknown non-CCU device coverage poisons the proof, rather than being
      // ignored and accepting a larger payload domain than the declared collective domain.
      if (ranks.empty()) coverage[it->second].insert(-1);
      coverage[it->second].insert(ranks.begin(), ranks.end());
      if (ccu) buffers[call->op_->name_].insert(it->second);
    }
  }
  const Chips& chips_;
  std::unordered_map<const Var*, const Var*> windows_;
  std::unordered_map<const Var*, int64_t> constants_;
  std::unordered_map<const Var*, std::set<int64_t>> loop_devices_;
  int loop_depth_ = 0;
  int unsupported_depth_ = 0;
};

// Host dispatch signatures stay unchanged. Nested device-side CHIP calls
// remain outside the qualified stage-1 domain-binding contract.
class CcuDispatchChecker : public IRVisitor {
 public:
  explicit CcuDispatchChecker(const Chips& chips) : chips_(chips) {}

 protected:
  void VisitExpr_(const CallPtr& op) override {
    Check(op);
    IRVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SubmitPtr& op) override {
    Check(op);
    IRVisitor::VisitExpr_(op);
  }

 private:
  template <typename T>
  void Check(const T& call) const {
    CHECK_SPAN(!chips_.count(call->op_->name_), call->span_)
        << "CCU CHIP callable requires a supported Host window dispatch; "
           "nested CHIP/InCore calls are unsupported";
  }
  const Chips& chips_;
};

}  // namespace

ProgramPtr LowerCcuCollectives(const ProgramPtr& program) {
  Chips chips;
  std::map<std::string, std::set<int64_t>> rank_counts;
  Names names;
  names.VisitProgram(program);
  for (const auto& [gvar, func] : program->functions_) {
    CcuFinder finder(func);
    finder.VisitStmt(func->body_);
    if (!finder.found) continue;
    // Re-running after ctx materialization would require remapping the suffix;
    // normal reruns are a no-op because public CCU ops have already disappeared.
    CHECK_SPAN(std::none_of(func->params_.begin(), func->params_.end(),
                            [](const auto& param) { return As<CommCtxType>(param->GetType()) != nullptr; }),
               func->span_)
        << "LowerL2TensorCollectives must run before MaterializeDistTensorCtx for CCU allgather";
    chips.insert(func->name_);
    rank_counts[func->name_] = std::move(finder.rank_counts);
  }
  if (chips.empty()) return program;

  std::map<std::string, FunctionPtr> kernels;
  auto result = MutableCopy(program);
  for (const auto& [gvar, func] : program->functions_) {
    auto updated = MutableCopy(func);
    if (chips.count(func->name_)) {
      CcuLowerer lowerer(&names, &kernels);
      updated->body_ = lowerer.Lower(func->body_);
    }
    if (IsOrchAt(func, Level::HOST)) {
      HostBindings bindings(chips);
      bindings.VisitStmt(func->body_);
      for (const auto& [callee, devices] : bindings.devices) {
        for (int64_t nranks : rank_counts.at(callee)) {
          if (nranks == 2) {
            CHECK_SPAN(devices.size() == 2, func->span_)
                << "CCU collective requires a static two-device domain";
          } else {
            CHECK_SPAN(devices.size() == static_cast<size_t>(nranks), func->span_)
                << "CCU collective Host domain size must match target rank count " << nranks;
          }
        }
        for (const Var* buffer : bindings.buffers.at(callee)) {
          CHECK_SPAN(bindings.coverage.at(buffer) == devices, func->span_)
              << "CCU collective supports one statically bound domain per callable; window device coverage "
                 "differs";
        }
      }
    } else {
      CcuDispatchChecker checker(chips);
      checker.VisitStmt(func->body_);
    }
    result->functions_[gvar] = updated;
  }
  for (const auto& [name, kernel] : kernels) {
    auto gvar = std::make_shared<GlobalVar>(name);
    CHECK_SPAN(!result->functions_.count(gvar), kernel->span_)
        << "CCU builtin name collides with existing function: " << name;
    result->functions_.emplace(gvar, kernel);
  }
  return result;
}

}  // namespace pypto::ir
