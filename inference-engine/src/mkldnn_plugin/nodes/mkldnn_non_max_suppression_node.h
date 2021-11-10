// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/nms_kernel.h"

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

class MKLDNNNonMaxSuppressionNode : public MKLDNNNode {
public:
    MKLDNNNonMaxSuppressionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() = default;
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index) :
                score(_score), batch_index(_batch_index), class_index(_class_index), box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

    float intersectionOverUnion(const float *boxesI, const float *boxesJ);

    void nmsWithSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                          const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes);

    void nmsWithoutSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                             const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes);

    void executeDynamicImpl(mkldnn::stream strm) override;

    bool needShapeInfer() const override { return false; }
    void prepareParams() override;

private:
    // input
    enum {
        NMS_BOXES,
        NMS_SCORES,
        NMS_MAXOUTPUTBOXESPERCLASS,
        NMS_IOUTHRESHOLD,
        NMS_SCORETHRESHOLD,
        NMS_SOFTNMSSIGMA,
    };

    // output
    enum {
        NMS_SELECTEDINDICES,
        NMS_SELECTEDSCORES,
        NMS_VALIDOUTPUTS
    };

    NMSBoxEncodeType boxEncodingType = NMSBoxEncodeType::CORNER;
    bool sortResultDescending = true;

    size_t numBatches = 0;
    size_t numBoxes = 0;
    size_t numClasses = 0;

    size_t maxOutputBoxesPerClass = 0lu;
    float iouThreshold = 0.0f;
    float scoreThreshold = 0.0f;
    float softNMSSigma = 0.0f;
    float scale = 1.f;
    // control placeholder for NMS in new opset.
    bool isSoftSuppressedByIOU = true;

    std::string errorPrefix;

    std::vector<std::vector<size_t>> numFiltBox;
    const std::string inType = "input", outType = "output";

    void checkPrecision(const Precision& prec, const std::vector<Precision>& precList, const std::string& name, const std::string& type);
    void check1DInput(const Shape& shape, const std::vector<Precision>& precList, const std::string& name, const size_t port);
    void checkOutput(const Shape& shape, const std::vector<Precision>& precList, const std::string& name, const size_t port);

    void createJitKernel();
    std::shared_ptr<jit_uni_nms_kernel> nms_kernel;
};

}  // namespace MKLDNNPlugin
