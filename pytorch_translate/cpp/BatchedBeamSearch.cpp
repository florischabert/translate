#include "BatchedBeamSearch.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <regex>

#include <caffe2/core/common.h>
#include <caffe2/core/logging.h>

#include "Dictionary.h"

namespace pytorch {
namespace translate {

BatchedBeamSearch::BatchedBeamSearch(
    const std::string& encoderModel,
    const std::string& decoderStepModel,
    int beamSize)
    : beamSize_(beamSize) {
  encoder_workspace_.reset(new ::caffe2::Workspace());
  encoder_.reset(new ::pytorch::translate::DbPredictor(
      encoderModel, encoder_workspace_.get()));
  decoder_workspace_.reset(new ::caffe2::Workspace());
  decoderStep_.reset(new ::pytorch::translate::DbPredictor(
      decoderStepModel, decoder_workspace_.get()));
}

BeamSearchOutput BatchedBeamSearch::beamSearch(
    const std::vector<int>& numberizedInput,
    int maxOutputSeqLen,
    bool reverseSource) {
  // first element of result vectors corresponds to input (unused)
  std::vector<std::vector<int>> tokenBeamList(1, std::vector<int>(beamSize_));
  std::vector<std::vector<float>> scoreBeamList(
      1, std::vector<float>(beamSize_));
  std::vector<std::vector<int>> prevIndexBeamList(
      1, std::vector<int>(beamSize_));
  std::vector<std::vector<std::vector<float>>> attentionWeightsBeamList(
      1,
      std::vector<std::vector<float>>(
          beamSize_, std::vector<float>(numberizedInput.size())));

  // Create tensor of numberizedInput
  auto inputBlob = caffe2::make_unique<caffe2::Blob>();
  caffe2::TensorCPU* inputTensor = BlobGetMutableTensor(inputBlob.get(), caffe2::CPU);
  inputTensor->Resize(numberizedInput.size(), 1);
  auto* inputPointer = inputTensor->mutable_data<long>();

  auto inputPointerIterator = inputPointer;
  if (reverseSource) {
    for (auto it = numberizedInput.rbegin(); it != numberizedInput.rend();
         ++it) {
      *inputPointerIterator++ = *it;
    }
  } else {
    for (auto it = numberizedInput.begin(); it != numberizedInput.end(); ++it) {
      *inputPointerIterator++ = *it;
    }
  }

  // Create tensor encoderLen
  auto encoderLenBlob = caffe2::make_unique<caffe2::Blob>();
  caffe2::TensorCPU* encoderLenTensor =
      BlobGetMutableTensor(encoderLenBlob.get(), caffe2::CPU);
  encoderLenTensor->Resize(1);
  auto* encoderLenPointer = encoderLenTensor->mutable_data<int>();
  encoderLenPointer[0] = numberizedInput.size();

  TensorMap inputMap;
  inputMap.emplace("encoder_inputs", inputTensor->Alias());
  inputMap.emplace("encoder_lengths", encoderLenTensor->Alias());

  TensorMap encoderOutputMap;
  CAFFE_ENFORCE((*encoder_)(inputMap, &encoderOutputMap));

  TensorMap stepInputMap = prepareInitialNextInputStepMap(
      encoder_->output_names(), encoderOutputMap);

  for (int decoderStep = 0; decoderStep <= maxOutputSeqLen; ++decoderStep) {
    TensorMap stepOutputMap;
    CAFFE_ENFORCE((*decoderStep_)(stepInputMap, &stepOutputMap));

    std::vector<long> bestTokensLong =
        tensorToVector1D<long>(stepOutputMap["best_tokens_indices"]);
    tokenBeamList.emplace_back(
        std::vector<int>(bestTokensLong.begin(), bestTokensLong.end()));

    std::vector<long> prevIndicesLong =
        tensorToVector1D<long>(stepOutputMap["prev_hypos_indices"]);
    prevIndexBeamList.emplace_back(
        std::vector<int>(prevIndicesLong.begin(), prevIndicesLong.end()));

    scoreBeamList.emplace_back(
        tensorToVector1D<float>(stepOutputMap["best_scores"]));
    attentionWeightsBeamList.emplace_back(
        tensorToVector2D<float>(stepOutputMap["attention_weights_average"]));

    stepInputMap = prepareNextInputStepMap(
        encoder_->output_names(),
        decoderStep_->output_names(),
        encoderOutputMap,
        stepOutputMap,
        decoderStep + 1);
  }

  BeamSearchOutput output(
      maxOutputSeqLen,
      tokenBeamList,
      scoreBeamList,
      prevIndexBeamList,
      attentionWeightsBeamList);
  return output;
}

TensorMap BatchedBeamSearch::prepareInitialNextInputStepMap(
    const std::vector<std::string>& encoderOutputNames,
    const TensorMap& encoderOutputMap) {
  std::regex encoderOutputRegex(
      "encoder_output_([0-9]+)", std::regex_constants::extended);
  std::regex initialStateRegex(
      "initial_state_([0-9]+)", std::regex_constants::extended);
  std::smatch regexMatch;
  TensorMap initialInputStepMap;
  for (const std::string& encoderOutputName : encoderOutputNames) {
    if (std::regex_search(encoderOutputName, regexMatch, encoderOutputRegex)) {
      initialInputStepMap["fixed_input_" + regexMatch.str(1)] =
          encoderOutputMap.at(encoderOutputName).Alias();
    } else if (std::regex_search(
                   encoderOutputName, regexMatch, initialStateRegex)) {
      initialInputStepMap["state_input_" + regexMatch.str(1)] =
          encoderOutputMap.at(encoderOutputName).Alias();
    } else if (encoderOutputName == "possible_translation_tokens") {
      initialInputStepMap["possible_translation_tokens"] =
          encoderOutputMap.at(encoderOutputName).Alias();
    } else {
      throw std::runtime_error(
          "Encoder output blob names should match "
          "encoder_output_([0-9]+) or initial_state_([0-9]+) or "
          "possible_translation_tokens - instead, found name: " +
          encoderOutputName);
    }
  }

  auto initialTimestepBlob = caffe2::make_unique<caffe2::Blob>();
  auto* initialTimestepTensor =
      BlobGetMutableTensor(initialTimestepBlob.get(), caffe2::CPU);
  initialTimestepTensor->Resize(1);
  initialTimestepTensor->mutable_data<int>()[0] = 0;

  auto initialPrevtokenBlob = caffe2::make_unique<caffe2::Blob>();
  auto* initialPrevtokenTensor =
      BlobGetMutableTensor(initialPrevtokenBlob.get(), caffe2::CPU);
  initialPrevtokenTensor->Resize(1);
  initialPrevtokenTensor->mutable_data<int>()[0] = kEosId;

  auto initialPrevScoresBlob = caffe2::make_unique<caffe2::Blob>();
  auto* initialPrevScoresTensor =
      BlobGetMutableTensor(initialPrevScoresBlob.get(), caffe2::CPU);
  initialPrevScoresTensor->Resize(1);
  initialPrevScoresTensor->mutable_data<float>()[0] = 0.0;

  initialInputStepMap["timestep"] = initialTimestepTensor->Alias();
  initialInputStepMap["prev_tokens"] = initialPrevtokenTensor->Alias();
  initialInputStepMap["prev_scores"] = initialPrevScoresTensor->Alias();

  return initialInputStepMap;
}

TensorMap BatchedBeamSearch::prepareNextInputStepMap(
    const std::vector<std::string>& encoderOutputNames,
    const std::vector<std::string>& stepOutputNames,
    TensorMap& encoderOutputMap,
    const TensorMap& stepOutputMap,
    int timeStep) {
  std::regex encoderOutputRegex(
      "encoder_output_([0-9]+)", std::regex_constants::extended);
  std::regex stepOutputRegex(
      "state_output_([0-9]+)", std::regex_constants::extended);
  std::smatch regexMatch;
  TensorMap inputStepMap;

  if (timeStep == 1) {
    // Encoder outputs tiled in place from shape (max_src_len, 1, H)
    // to (max_src_len, beam_size, H) on sencond step (used by all future steps)
    for (const std::string& encoderOutputName : encoderOutputNames) {
      if (std::regex_search(
              encoderOutputName, regexMatch, encoderOutputRegex)) {
        caffe2::TensorCPU &untiledTensor = encoderOutputMap[encoderOutputName];

        auto tiledEncoderOutputsBlob = caffe2::make_unique<caffe2::Blob>();
        caffe2::TensorCPU* tiledEncoderOutputTensor =
            BlobGetMutableTensor(tiledEncoderOutputsBlob.get(), caffe2::CPU);
        auto sourceLength = untiledTensor.sizes()[0];
        auto hiddenSize = untiledTensor.sizes()[2];
        tiledEncoderOutputTensor->Resize(sourceLength, beamSize_, hiddenSize);
        auto* tiledEncoderOutputPointer =
            tiledEncoderOutputTensor->mutable_data<float>();
        auto* untiledPointer = untiledTensor.data<float>();
        for (int i = 0; i < sourceLength; ++i) {
          for (int j = 0; j < beamSize_; ++j) {
            auto tiledIndex = i * beamSize_ * hiddenSize + j * hiddenSize;
            auto untiledIndex = i * hiddenSize;
            memcpy(
                tiledEncoderOutputPointer + tiledIndex,
                untiledPointer + untiledIndex,
                sizeof(float) * hiddenSize);
          }
        }
        encoderOutputMap[encoderOutputName] = tiledEncoderOutputTensor->Alias();
      }
    }
  }

  for (const std::string& encoderOutputName : encoderOutputNames) {
    if (std::regex_search(encoderOutputName, regexMatch, encoderOutputRegex)) {
      inputStepMap["fixed_input_" + regexMatch.str(1)] =
          encoderOutputMap[encoderOutputName].Alias();
    } else if (encoderOutputName == "possible_translation_tokens") {
      inputStepMap["possible_translation_tokens"] =
          encoderOutputMap[encoderOutputName].Alias();
    }
  }
  for (const std::string& stepOutputName : stepOutputNames) {
    if (std::regex_search(stepOutputName, regexMatch, stepOutputRegex)) {
      inputStepMap["state_input_" + regexMatch.str(1)] =
          stepOutputMap.at(stepOutputName).Alias();
    }
  }

  auto timestepBlob = caffe2::make_unique<caffe2::Blob>();
  auto* timestepTensor = BlobGetMutableTensor(timestepBlob.get(), caffe2::CPU);
  timestepTensor->Resize(1);
  timestepTensor->mutable_data<int>()[0] = timeStep;

  inputStepMap["timestep"] = timestepTensor->Alias();
  inputStepMap["prev_tokens"] = stepOutputMap.at("best_tokens_indices").Alias();
  inputStepMap["prev_scores"] = stepOutputMap.at("best_scores").Alias();

  return inputStepMap;
}

} // namespace translate
} // namespace pytorch
