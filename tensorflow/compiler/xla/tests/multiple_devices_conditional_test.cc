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

// This demonstrates how to use hlo_test_base to create textual IR based
// testcases.

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using absl::nullopt;

class MultipleDevicesConditionalTest : public HloTestBase {};

TEST_F(MultipleDevicesConditionalTest, Axpy) {
  const string& hlo_string = R"(
HloModule axpy_module:
ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0.0001}));
}

TEST_F(MultipleDevicesConditionalTest, Tuple) {
  const string& hlo_string = R"(
HloModule TupleCreate_module:
ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, nullopt));
}

XLA_TEST_F(MultipleDevicesConditionalTest, MultipleOperands) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p0 = f32[3] parameter(0)
    p1 = f32[2] parameter(1)
    ROOT crs = (f32[3], f32[2]) all-reduce(p0, p1), to_apply=add
  })";
  auto module =
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest())
          .ValueOrDie();
  auto literal0 = LiteralUtil::CreateR1<float>({1, 2, 3});
  auto literal1 = LiteralUtil::CreateR1<float>({10, 20});
  EXPECT_EQ(LiteralUtil::MakeTuple({&literal0, &literal1}),
            ExecuteAndTransfer(std::move(module), {&literal0, &literal1}));
}

}  // namespace
}  // namespace xla
