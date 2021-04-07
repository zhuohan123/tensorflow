/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <random>
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class ConditionalOpTest : public ClientLibraryTestBase {
 protected:
  XlaComputation CreateR0ConstantComputation(float value) {
    XlaBuilder builder("Constant");
    Parameter(&builder, 0, empty_tuple_, "tuple");
    ConstantR0<float>(&builder, value);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0IdentityComputation() {
    XlaBuilder builder("Identity");
    Parameter(&builder, 0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateCeilComputation(const Shape& shape) {
    XlaBuilder builder("Ceil");
    auto param = Parameter(&builder, 0, shape, "param");
    Ceil(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0CeilComputation() {
    return CreateCeilComputation(r0f32_);
  }

  XlaComputation CreateR1CeilComputation() {
    return CreateCeilComputation(r1s2f32_);
  }

  XlaComputation CreateFloorComputation(const Shape& shape) {
    XlaBuilder builder("Floor");
    auto param = Parameter(&builder, 0, shape, "param");
    Floor(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0FloorComputation() {
    return CreateFloorComputation(r0f32_);
  }

  XlaComputation CreateR1FloorComputation() {
    return CreateFloorComputation(r1s2f32_);
  }

  XlaComputation CreateTupleCeilComputation(const string& computation_name,
                                            const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    auto x_ceil = Ceil(x);
    auto y_ceil = Ceil(y);
    Tuple(&builder, {x_ceil, y_ceil});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleCeilComputation() {
    return CreateTupleCeilComputation("CeilR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleCeilComputation() {
    return CreateTupleCeilComputation("CeilR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleFloorComputation(const string& computation_name,
                                             const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    auto x_floor = Floor(x);
    auto y_floor = Floor(y);
    Tuple(&builder, {x_floor, y_floor});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleFloorComputation() {
    return CreateTupleFloorComputation("FloorR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleFloorComputation() {
    return CreateTupleFloorComputation("FloorR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleAddComputation(const string& computation_name,
                                           const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleAddComputation() {
    return CreateTupleAddComputation("AddR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleAddComputation() {
    return CreateTupleAddComputation("AddR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleSubComputation(const string& computation_name,
                                           const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    Sub(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleSubComputation() {
    return CreateTupleSubComputation("SubR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleSubComputation() {
    return CreateTupleSubComputation("SubR1", tuple_2_r1s2f32_);
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r1s2f32_ = ShapeUtil::MakeShape(F32, {2});
  Shape tuple_2_r0f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});
  Shape tuple_2_r1s2f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeShape(F32, {2})});
  Shape empty_tuple_ = ShapeUtil::MakeTupleShape({});
  ErrorSpec error_spec_{0.001};
};

// Test true and false computations that take in 2 array parameters and
// predicate is true.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayTrueBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR1<float>(&builder, {24.0f, 56.0f});
  auto operand2 = ConstantR1<float>(&builder, {10.0f, 11.0f});
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {34.0f, 67.0f}, {pred_arg.get()},
                             error_spec_);
}

}  // namespace
}  // namespace xla
