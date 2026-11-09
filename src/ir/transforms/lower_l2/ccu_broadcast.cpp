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

#include "src/ir/transforms/lower_l2/ccu_broadcast.h"

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
#include "pypto/ir/transforms/utils/ccu_broadcast_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"

namespace pypto::ir {
bool IsCcuBroadcast(const CallPtr& call) {
  return IsOp(call, "pld.tensor.broadcast") && call->GetKwarg<std::string>("engine", "default") == "ccu";
}

namespace {

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

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (IsCcuBroadcast(call)) {
      CHECK_SPAN(IsOrchAt(func_, Level::CHIP), call->span_)
          << "pld.tensor.broadcast engine='ccu' is supported only in CHIP orchestration, not HOST/InCore";
      CHECK_SPAN(depth_ == 0, call->span_) << "CHIP CCU broadcast requires a straight-line sequence; loops "
                                              "and conditional calls are unsupported";
      found = true;
    }
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    CHECK_SPAN(
        !IsOp(submit, "pld.tensor.broadcast") || submit->GetKwarg<std::string>("engine", "default") != "ccu",
        submit->span_)
        << "CCU broadcast direct-op Submit is unsupported; submit a CHIP callable instead";
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

// Validation and specialization are deliberately an allowlist: two ranks,
// FP32, static positive packed ND tensors with no partial valid region.
std::string ValidateView(const DistributedTensorTypePtr& type, const Span& span) {
  CHECK_SPAN(type && !type->shape_.empty(), span) << "CCU broadcast requires a nonempty static shape";
  std::string key;
  int64_t stride = 1;
  const auto& view = type->tensor_view_;
  CHECK_SPAN(!view || view->layout == TensorLayout::ND, span) << "CCU broadcast requires contiguous ND views";
  CHECK_SPAN(!view || view->stride.empty() || view->stride.size() == type->shape_.size(), span)
      << "CCU broadcast requires full contiguous strides";
  CHECK_SPAN(!view || view->valid_shape.empty() || view->valid_shape.size() == type->shape_.size(), span)
      << "CCU broadcast requires a full valid view";
  for (size_t i = type->shape_.size(); i-- > 0;) {
    auto extent = As<ConstInt>(type->shape_[i]);
    CHECK_SPAN(extent && extent->value_ > 0, span) << "CCU broadcast requires static positive dimensions";
    if (view && !view->stride.empty()) {
      auto actual = As<ConstInt>(view->stride[i]);
      CHECK_SPAN(actual && actual->value_ == stride, span) << "CCU broadcast requires contiguous ND strides";
    }
    if (view && !view->valid_shape.empty()) {
      auto valid = As<ConstInt>(view->valid_shape[i]);
      CHECK_SPAN(valid && valid->value_ == extent->value_, span)
          << "CCU broadcast requires a full valid view";
    }
    CHECK_SPAN(extent->value_ <= std::numeric_limits<int64_t>::max() / stride, span)
        << "CCU broadcast element count exceeds INT64_MAX";
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
    if (!IsCcuBroadcast(call)) return call;
    auto target = As<DistributedTensorType>(call->args_[0]->GetType());
    auto signal = As<DistributedTensorType>(call->args_[1]->GetType());
    auto shape_key = ValidateView(target, call->span_);
    ValidateView(signal, call->span_);
    CHECK_SPAN(target->dtype_ == DataType::FP32, call->span_) << "CCU broadcast supports only FP32 target";
    CHECK_SPAN(signal->shape_.size() == 2 && As<ConstInt>(signal->shape_[0])->value_ == 2 &&
                   As<ConstInt>(signal->shape_[1])->value_ == 1,
               call->span_)
        << "CCU broadcast supports only two ranks with signal shape [2, 1]";
    auto root = call->GetKwarg<int>("root");
    CHECK_SPAN(root >= 0 && root < 2, call->span_) << "CCU broadcast root must satisfy 0 <= root < 2";
    const auto* handler = PassContext::Current()->GetBackendHandler();
    CHECK_SPAN(handler && handler->SupportCcu(), call->span_)
        << "CCU broadcast is unsupported by this backend; compiler lowering requires Ascend950";

    // CCU entry identity excludes shape: element count is a runtime argument.
    // Root/dtype/rank specialization identifies the CCU function, while shape
    // only specializes the AIV wrapper's tensor signature and return alias.
    const auto entry = "dtype_cpp=float,nranks_cpp=2,root_cpp=" + std::to_string(root);
    auto [token, inserted] = tokens_.try_emplace(std::string(kCcuBroadcastTemplate) + ":" + entry);
    if (inserted) {
      auto shape = std::make_shared<MakeTuple>(std::vector<ExprPtr>{Index(1, call->span_)}, call->span_);
      auto alloc = OpRegistry::GetInstance().Create(
          "tensor.create", {shape}, {{"dtype", DataType::UINT8}, {"layout", TensorLayout::ND}}, call->span_);
      token->second = std::make_shared<Var>(names_->Fresh("_ccu_order_"), alloc->GetType(), call->span_);
      allocations_.push_back(std::make_shared<AssignStmt>(token->second, alloc, call->span_));
    }
    auto name = "__builtin_broadcast_ccu__fp32_p2_r" + std::to_string(root) + shape_key;
    if (!kernels_->count(name)) {
      std::vector<VarPtr> params;
      // Canonical packed signatures do not capture the first caller's window
      // identity or memref. The shape key prevents incompatible return aliases.
      params.push_back(std::make_shared<Var>(
          "target", std::make_shared<DistributedTensorType>(target->shape_, target->dtype_), call->span_));
      params.push_back(std::make_shared<Var>(
          "signal", std::make_shared<DistributedTensorType>(signal->shape_, signal->dtype_), call->span_));
      params.push_back(std::make_shared<Var>("order", OrderType(call->span_), call->span_));
      auto body = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{params[0]}, call->span_);
      std::vector<std::pair<std::string, std::any>> attrs{
          {kAttrBuiltinTemplateDir, std::string(kCcuBroadcastTemplate)}, {kAttrBuiltinTemplateVars, entry}};
      (*kernels_)[name] =
          std::make_shared<Function>(name, params, std::vector<ParamDirection>(3, ParamDirection::InOut),
                                     std::vector<TypePtr>{params[0]->GetType()}, body, call->span_,
                                     FunctionType::AIV, std::nullopt, std::nullopt, attrs);
    }
    // The explicit single-block launch is compiler-owned, never an API option.
    std::vector<std::pair<std::string, std::any>> attrs{{kAttrCoreNum, Index(1, call->span_)}};
    return std::make_shared<Call>(std::make_shared<GlobalVar>(name),
                                  std::vector<ExprPtr>{call->args_[0], call->args_[1], token->second},
                                  std::vector<std::pair<std::string, std::any>>{}, attrs, call->GetType(),
                                  call->span_);
  }

 private:
  // Each Lowerer instance owns one CHIP invocation's tokens. A shared InOut
  // tensor creates a WAW chain only for repeated uses of the same CCU entry.
  // No remote access, Host parameter, communication window or ctx is needed.
  Names* names_;
  std::map<std::string, VarPtr> tokens_;
  std::vector<StmtPtr> allocations_;
  std::map<std::string, FunctionPtr>* kernels_;
};

// A deliberately local binding analysis: direct window/SSA aliases, constant
// devices or one static two-device dispatch loop. No cross-function provenance
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
    if (start && stop && step && *start >= 0 && *step == 1 && *stop - *start == 2)
      loop_devices_[op->loop_var_.get()] = {*start, *start + 1};
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
          << "CCU CHIP dispatch requires constant devices or one static two-device rank loop; "
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
      // ignored and accepting a larger payload domain than the qualified pair.
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

ProgramPtr LowerCcuBroadcast(const ProgramPtr& program) {
  Chips chips;
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
        << "LowerL2TensorCollectives must run before MaterializeDistTensorCtx for CCU broadcast";
    chips.insert(func->name_);
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
        CHECK_SPAN(devices.size() == 2, func->span_) << "CCU broadcast requires a static two-device domain";
        for (const Var* buffer : bindings.buffers.at(callee)) {
          CHECK_SPAN(bindings.coverage.at(buffer) == devices, func->span_)
              << "CCU broadcast supports one statically bound domain per callable; window device coverage "
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
