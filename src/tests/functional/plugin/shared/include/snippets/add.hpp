// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        InputShape,                  // Input 1 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> AddParams;

typedef std::tuple<
        std::vector<InputShape>,     // Input 0, Input 1 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> AddParamsPair;

typedef std::tuple<
        InputShape,                  // Input 0 Shape
        ov::element::Type,           // Element type
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> AddConstParams;

class Add : public testing::WithParamInterface<ov::test::snippets::AddParams>,
            virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddParams> obj);

protected:
    void SetUp() override;
};

class AddConst : public testing::WithParamInterface<ov::test::snippets::AddConstParams>,
                 virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddConstParams> obj);
protected:
    void SetUp() override;
};

class AddRollConst : public AddConst {
protected:
    void SetUp() override;
};

// repack AddPair input shapes into shape vector to cover some cases easier
class AddPair : public testing::WithParamInterface<ov::test::snippets::AddParamsPair>,
                virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddParamsPair> obj);
protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov