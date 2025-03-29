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

/// \file GPUTPCNNClusterizerHost.cxx
/// \author Christian Sonnabend

#include <CommonUtils/StringUtils.h>

#include "GPUTPCNNClusterizerHost.h"
#include "GPUTPCNNClusterizer.h"
#include "CCDB/CcdbApi.h"
#include "GPUSettings.h"
#include "ML/3rdparty/GPUORTFloat16.h"

using namespace o2::gpu;

GPUTPCNNClusterizerHost::GPUTPCNNClusterizerHost(const GPUSettingsProcessingNNclusterizer& settings, int32_t deviceId)
{
  init(settings, deviceId);
}

void GPUTPCNNClusterizerHost::loadFromCCDB(std::map<std::string, std::string> settings)
{
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(settings["nnCCDBURL"]);

  metadata["inputDType"] = settings["inputDType"];
  metadata["outputDType"] = settings["outputDType"];
  metadata["nnCCDBEvalType"] = settings["nnCCDBEvalType"];         // classification_1C, classification_2C, regression_1C, regression_2C
  metadata["nnCCDBWithMomentum"] = settings["nnCCDBWithMomentum"]; // 0, 1 -> Only for regression model
  metadata["nnCCDBLayerType"] = settings["nnCCDBLayerType"];       // FC, CNN
  if (settings["nnCCDBInteractionRate"] != "" && std::stoi(settings["nnCCDBInteractionRate"]) > 0) {
    metadata["nnCCDBInteractionRate"] = settings["nnCCDBInteractionRate"];
  }
  if (settings["nnCCDBBeamType"] != "") {
    metadata["nnCCDBBeamType"] = settings["nnCCDBBeamType"];
  }

  bool retrieveSuccess = ccdbApi.retrieveBlob(settings["nnCCDBPath"], ".", metadata, 1, false, settings["outputFile"]);
  // headers = ccdbApi.retrieveHeaders(settings["nnPathCCDB"], metadata, 1); // potentially needed to init some local variables

  if (retrieveSuccess) {
    LOG(info) << "Network " << settings["nnCCDBPath"] << " retrieved from CCDB, stored at " << settings["outputFile"];
  } else {
    LOG(error) << "Failed to retrieve network from CCDB";
  }
}

void GPUTPCNNClusterizerHost::init(const GPUSettingsProcessingNNclusterizer& settings, int32_t deviceId)
{
  std::string class_model_path = settings.nnClassificationPath, reg_model_path = settings.nnRegressionPath;
  std::vector<std::string> reg_model_paths;

  if (settings.nnLoadFromCCDB) {
    std::map<std::string, std::string> ccdbSettings = {
      {"nnCCDBURL", settings.nnCCDBURL},
      {"nnCCDBPath", settings.nnCCDBPath},
      {"inputDType", settings.nnInferenceInputDType},
      {"outputDType", settings.nnInferenceOutputDType},
      {"nnCCDBWithMomentum", std::to_string(settings.nnCCDBWithMomentum)},
      {"nnCCDBBeamType", settings.nnCCDBBeamType},
      {"nnCCDBInteractionRate", std::to_string(settings.nnCCDBInteractionRate)}};

    std::string nnFetchFolder = "";
    std::vector<std::string> fetchMode = o2::utils::Str::tokenize(settings.nnCCDBFetchMode, ':');
    std::map<std::string, std::string> networkRetrieval = ccdbSettings;

    if (fetchMode[0] == "c1") {
      networkRetrieval["nnCCDBLayerType"] = settings.nnCCDBClassificationLayerType;
      networkRetrieval["nnCCDBEvalType"] = "classification_c1";
      networkRetrieval["outputFile"] = nnFetchFolder + "net_classification_c1.onnx";
      loadFromCCDB(networkRetrieval);
    } else if (fetchMode[0] == "c2") {
      networkRetrieval["nnCCDBLayerType"] = settings.nnCCDBClassificationLayerType;
      networkRetrieval["nnCCDBEvalType"] = "classification_c2";
      networkRetrieval["outputFile"] = nnFetchFolder + "net_classification_c2.onnx";
      loadFromCCDB(networkRetrieval);
    }
    class_model_path = networkRetrieval["outputFile"]; // Setting the proper path from the where the models will be initialized locally

    networkRetrieval["nnCCDBLayerType"] = settings.nnCCDBRegressionLayerType;
    networkRetrieval["nnCCDBEvalType"] = "regression_c1";
    networkRetrieval["outputFile"] = nnFetchFolder + "net_regression_c1.onnx";
    loadFromCCDB(networkRetrieval);
    reg_model_path = networkRetrieval["outputFile"];
    if (fetchMode[1] == "r2") {
      networkRetrieval["nnCCDBLayerType"] = settings.nnCCDBRegressionLayerType;
      networkRetrieval["nnCCDBEvalType"] = "regression_c2";
      networkRetrieval["outputFile"] = nnFetchFolder + "net_regression_c2.onnx";
      loadFromCCDB(networkRetrieval);
      reg_model_path += ":", networkRetrieval["outputFile"];
    }
  }

  OrtOptions = {
    {"model-path", class_model_path},
    {"device", settings.nnInferenceDevice},
    {"device-id", std::to_string(deviceId)},
    {"allocate-device-memory", std::to_string(settings.nnInferenceAllocateDevMem)},
    {"intra-op-num-threads", std::to_string(settings.nnInferenceIntraOpNumThreads)},
    {"inter-op-num-threads", std::to_string(settings.nnInferenceInterOpNumThreads)},
    {"enable-optimizations", std::to_string(settings.nnInferenceEnableOrtOptimization)},
    {"enable-profiling", std::to_string(settings.nnInferenceOrtProfiling)},
    {"profiling-output-path", settings.nnInferenceOrtProfilingPath},
    {"logging-level", std::to_string(settings.nnInferenceVerbosity)}};

  model_class.init(OrtOptions);

  reg_model_paths = o2::utils::Str::tokenize(reg_model_path, ':');

  if (!settings.nnClusterizerUseCfRegression) {
    if (model_class.getNumOutputNodes()[0][1] == 1 || reg_model_paths.size() == 1) {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
    } else {
      OrtOptions["model-path"] = reg_model_paths[0];
      model_reg_1.init(OrtOptions);
      OrtOptions["model-path"] = reg_model_paths[1];
      model_reg_2.init(OrtOptions);
    }
  }
}

void GPUTPCNNClusterizerHost::initClusterizer(const GPUSettingsProcessingNNclusterizer& settings, GPUTPCNNClusterizer& clusterer)
{
  clusterer.nnClusterizerModelClassNumOutputNodes = model_class.getNumOutputNodes()[0][1];
  if (!settings.nnClusterizerUseCfRegression) {
    if (model_class.getNumOutputNodes()[0][1] == 1 || model_reg_2.isInitialized()) {
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
    } else {
      clusterer.nnClusterizerModelReg1NumOutputNodes = model_reg_1.getNumOutputNodes()[0][1];
      clusterer.nnClusterizerModelReg2NumOutputNodes = model_reg_2.getNumOutputNodes()[0][1];
    }
  }
}

void GPUTPCNNClusterizerHost::networkInference(o2::ml::OrtModel model, GPUTPCNNClusterizer& clustererNN, size_t size, float* output, int32_t dtype)
{
  if (dtype == 0) {
    model.inference<OrtDataType::Float16_t, float>(clustererNN.inputData16, size, output);
  } else {
    model.inference<float, float>(clustererNN.inputData32, size, output);
  }
}
