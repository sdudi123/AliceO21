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

/// \file GPUTPCNNClusterizerHost.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZERHOST_H
#define O2_GPUTPCNNCLUSTERIZERHOST_H

#include <string>
#include <unordered_map>
#include <vector>
#include "ML/OrtInterface.h"

using namespace o2::ml;

namespace o2::OrtDataType
{
struct Float16_t;
}

namespace o2::gpu
{

class GPUTPCNNClusterizer;
struct GPUSettingsProcessingNNclusterizer;

class GPUTPCNNClusterizerHost
{
 public:
  GPUTPCNNClusterizerHost() = default;
  GPUTPCNNClusterizerHost(const GPUSettingsProcessingNNclusterizer& settings) { init(settings); }

  void init(const GPUSettingsProcessingNNclusterizer&);
  void initClusterizer(const GPUSettingsProcessingNNclusterizer&, GPUTPCNNClusterizer&);
  void loadFromCCDB(std::map<std::string, std::string>);

  void networkInference(o2::ml::OrtModel, GPUTPCNNClusterizer&, size_t, float*, int32_t);

  std::unordered_map<std::string, std::string> OrtOptions;
  o2::ml::OrtModel model_class, model_reg_1, model_reg_2; // For splitting clusters
  std::vector<bool> modelsUsed = {false, false, false}; // 0: class, 1: reg_1, 2: reg_2
  int32_t deviceId = -1;
  std::vector<std::string> reg_model_paths;

 private:
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
}; // class GPUTPCNNClusterizerHost

} // namespace o2::gpu

#endif
