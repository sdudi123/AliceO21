// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNNClusterizerInternals.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZERINTERNALS_H
#define O2_GPUTPCNNCLUSTERIZERINTERNALS_H

#include "ML/3rdparty/GPUORTFloat16.h"
#include "ML/OrtInterface.h"
#include "ChargePos.h"

using namespace o2::ml;

namespace o2::gpu
{

class GPUTPCNNClusterizerInternals
{
 public:
  int nnClusterizerSizeInputRow = 3;
  int nnClusterizerSizeInputPad = 3;
  int nnClusterizerSizeInputTime = 3;
  int nnClusterizerElementSize = -1;
  bool nnClusterizerAddIndexData = true;
  float nnClassThreshold = 0.16;
  bool nnSigmoidTrafoClassThreshold = 1;
  int nnClusterizerUseCfRegression = 0;
  int nnClusterizerBatchedMode = 1;
  int nnClusterizerVerbosity = 0;
  int nnClusterizerBoundaryFillValue = -1;
  int nnClusterizerDumpDigits = 0;
  int nnClusterizerApplyCfDeconvolution = 0;
  int nnClusterizerModelClassNumOutputNodes = -1;
  int nnClusterizerModelReg1NumOutputNodes = -1;
  int nnClusterizerModelReg2NumOutputNodes = -1;

  // Memory allocation for neural network
  uint class2_elements = 0;
  std::vector<float> inputData32;
  std::vector<OrtDataType::Float16_t> inputData16;
  std::vector<float> outputDataClass, modelProbabilities, outputDataReg1, outputDataReg2;

  std::vector<ChargePos> peakPositions;
  std::vector<std::vector<bool>> clusterFlags; // mSplitInTime, mSplitInPad. Techincally both flags are set in the same way -> ClusterAccumulator.cxx
  std::vector<float> centralCharges;

  std::unordered_map<std::string, std::string> OrtOptions;
  o2::ml::OrtModel model_class, model_reg_1, model_reg_2; // For splitting clusters
}; // class GPUTPCNNClusterizerInternals

} // namespace o2::gpu

#endif