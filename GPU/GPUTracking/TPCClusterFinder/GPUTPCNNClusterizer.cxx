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

/// \file GPUTPCNNClusterizer.cxx
/// \author Christian Sonnabend

#include "GPUTPCNNClusterizer.h"
#include "GPUTPCCFClusterizer.h"

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

// Defining individual thread functions for data filling, determining the class label and running the CF clusterizer
template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::runCfClusterizer>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  if ((clusterer.nnInternals)->outputDataClass[glo_idx] == 0) { // default clusterizer should not be called in batched mode due to mess-up with thread indices
    return;
  }
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAcc(clusterer));
  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
  o2::gpu::GPUTPCCFClusterizer::GPUSharedMemory smem_new;
  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, smem_new, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::fillInputNN>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  GPUTPCNNClusterizer::fillInputData(nBlocks, nThreads, iBlock, iThread, clusterer, dtype, batchStart);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::determineClass1Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  (clusterer.nnInternals)->outputDataClass[glo_idx + batchStart] = (int)((clusterer.nnInternals)->modelProbabilities[glo_idx] > (clusterer.nnInternals)->nnClassThreshold);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::determineClass2Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  uint elem_iterator = glo_idx * (clusterer.nnInternals)->model_class.getNumOutputNodes()[0][1];
  float current_max_prob = 0.f; // If the neural network doesn't contain the softmax as a last layer, the outputs can range in [-infty, infty]
  uint class_label = 0;
  for (float pIdx = elem_iterator; pIdx < elem_iterator + (clusterer.nnInternals)->model_class.getNumOutputNodes()[0][1]; pIdx++) {
    if (pIdx == elem_iterator) {
      current_max_prob = (clusterer.nnInternals)->modelProbabilities[pIdx];
    } else {
      class_label = ((clusterer.nnInternals)->modelProbabilities[pIdx] > current_max_prob ? pIdx : class_label);
    }
  }
  // uint class_label = std::distance(elem_iterator, std::max_element(elem_iterator, elem_iterator + (clusterer.nnInternals)->model_class.getNumOutputNodes()[0][1])); // Multiple outputs of the class network are the probabilities for each class. The highest one "wins"
  (clusterer.nnInternals)->outputDataClass[glo_idx + batchStart] = class_label;
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::publishClass1Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  if (glo_idx >= clusterer.mPmemory->counters.nClusters) {
    return;
  }
  GPUTPCNNClusterizer::publishClustersReg1(glo_idx, smem, clusterer, dtype, onlyMC, batchStart);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::publishClass2Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  if (glo_idx >= clusterer.mPmemory->counters.nClusters) {
    return;
  }
  GPUTPCNNClusterizer::publishClustersReg2(glo_idx, smem, clusterer, dtype, onlyMC, batchStart);
}

// Apply the neural network to the input data. Note: These are not GPU kernels. We let ONNX take care of that
void GPUTPCNNClusterizer::inferenceNetworkClass(processorType& clusterer, int8_t dtype, uint batch_idx)
{
  if (dtype == 0) {
    (clusterer.nnInternals)->modelProbabilities = (clusterer.nnInternals)->model_class.inference<OrtDataType::Float16_t, float>((clusterer.nnInternals)->inputData16);
  } else {
    (clusterer.nnInternals)->modelProbabilities = (clusterer.nnInternals)->model_class.inference<float, float>((clusterer.nnInternals)->inputData32);
  }
}

void GPUTPCNNClusterizer::inferenceNetworkReg1(processorType& clusterer, int8_t dtype)
{
  if (dtype == 0) {
    (clusterer.nnInternals)->outputDataReg1 = (clusterer.nnInternals)->model_reg_1.inference<OrtDataType::Float16_t, float>((clusterer.nnInternals)->inputData16);
  } else {
    (clusterer.nnInternals)->outputDataReg1 = (clusterer.nnInternals)->model_reg_1.inference<float, float>((clusterer.nnInternals)->inputData32);
  }
}

void GPUTPCNNClusterizer::inferenceNetworkReg2(processorType& clusterer, int8_t dtype)
{
  if (dtype == 0) {
    (clusterer.nnInternals)->outputDataReg2 = (clusterer.nnInternals)->model_reg_2.inference<OrtDataType::Float16_t, float>((clusterer.nnInternals)->inputData16);
  } else {
    (clusterer.nnInternals)->outputDataReg2 = (clusterer.nnInternals)->model_reg_2.inference<float, float>((clusterer.nnInternals)->inputData32);
  }
}

// THe following arithmetic is done because the network is trained with a split between IROC and OROC boundary
int GPUTPCNNClusterizer::padOffset(int row_ref, int row_current, const GPUTPCGeometry& geo)
{
  return (int)((geo.NPads(row_current) - geo.NPads(row_ref)) / 2);
}

int GPUTPCNNClusterizer::rowOffset(int row, int global_shift)
{
  return (row > 62 ? global_shift : 0);
}

bool GPUTPCNNClusterizer::isBoundary(int row, int pad, int global_shift, const GPUTPCGeometry& geo)
{
  if (pad < 0 || row < 0) { // Faster short-circuit
    return true;
  } else if (row < 63) {
    return (pad >= static_cast<int>(geo.NPads(row)));
  } else if (row < (63 + global_shift)) { // to account for the gap between IROC and OROC. Charge will be set to -1 in order to signal boundary to the neural network
    return true;
  } else if (row <= o2::tpc::constants::MAXGLOBALPADROW - 1 + global_shift) {
    return (pad >= static_cast<int>(geo.NPads(row - global_shift)));
  } else {
    return true;
  }
}

// Filling the input data for the neural network where there is no boundary
GPUd() void GPUTPCNNClusterizer::fillInputData(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, processorType& clusterer, int8_t dtype, uint batchStart)
{

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uint8_t> isPeakMap(clusterer.mPpeakMap);

  uint glo_idx = get_global_id(0);

  uint write_idx = glo_idx * (clusterer.nnInternals)->nnClusterizerElementSize; // Potential optimization: Either choose nnClusterizerBatchedMode as a power of 2 or calculate from threadId and blockId

  ChargePos peak = clusterer.mPfilteredPeakPositions[glo_idx + batchStart];
  int row = static_cast<int>(peak.row()), pad = static_cast<int>(peak.pad()), time = static_cast<int>(peak.time()); // Explicit casting to avoid conversion errors
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  (clusterer.nnInternals)->peakPositions[glo_idx] = peak;
  (clusterer.nnInternals)->centralCharges[glo_idx] = central_charge;

  int row_offset = GPUTPCNNClusterizer::rowOffset(row, (clusterer.nnInternals)->nnClusterizerSizeInputRow);
  GPUCA_UNROLL(U(), U());
  for (int r = -(clusterer.nnInternals)->nnClusterizerSizeInputRow; r <= (clusterer.nnInternals)->nnClusterizerSizeInputRow; r++) {
    bool is_row_boundary = ((row + r) > (o2::tpc::constants::MAXGLOBALPADROW - 1)) || ((row + r) < 0);
    int pad_offset = is_row_boundary ? 0 : GPUTPCNNClusterizer::padOffset(row, row + r, clusterer.Param().tpcGeometry);
    for (int p = -(clusterer.nnInternals)->nnClusterizerSizeInputPad + pad_offset; p <= (clusterer.nnInternals)->nnClusterizerSizeInputPad + pad_offset; p++) {
      bool is_boundary = is_row_boundary || GPUTPCNNClusterizer::isBoundary(row + r + row_offset, pad + p, (clusterer.nnInternals)->nnClusterizerSizeInputRow, clusterer.Param().tpcGeometry);
      for (int t = -(clusterer.nnInternals)->nnClusterizerSizeInputTime; t <= (clusterer.nnInternals)->nnClusterizerSizeInputTime; t++) {
        if (!is_boundary) {
          ChargePos tmp_pos(row + r, pad + p, time + t);
          if (r == 0 && !(clusterer.nnInternals)->clusterFlags[glo_idx][0] && std::abs(p) < 3 && std::abs(t) < 3 && p != 0 && t != 0) { // ordering is done for short circuit optimization
            (clusterer.nnInternals)->clusterFlags[glo_idx][0] = CfUtils::isPeak(isPeakMap[tmp_pos]);
            (clusterer.nnInternals)->clusterFlags[glo_idx][1] = (clusterer.nnInternals)->clusterFlags[glo_idx][0];
          }
          if (dtype == 0) {
            (clusterer.nnInternals)->inputData16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge);
          } else {
            (clusterer.nnInternals)->inputData32[write_idx] = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge;
          }
        } else {
          // Filling boundary just to make sure that no values are left unintentionally
          if (dtype == 0) {
            (clusterer.nnInternals)->inputData16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>((clusterer.nnInternals)->nnClusterizerBoundaryFillValue));
          } else {
            (clusterer.nnInternals)->inputData32[write_idx] = static_cast<float>((clusterer.nnInternals)->nnClusterizerBoundaryFillValue);
          }
        }
        write_idx++;
      }
    }
  }
  if ((clusterer.nnInternals)->nnClusterizerAddIndexData) {
    if (dtype == 0) {
      (clusterer.nnInternals)->inputData16[write_idx] = (OrtDataType::Float16_t)(clusterer.mISlice / 36.f);
      (clusterer.nnInternals)->inputData16[write_idx + 1] = (OrtDataType::Float16_t)(row / 152.f);
      (clusterer.nnInternals)->inputData16[write_idx + 2] = (OrtDataType::Float16_t)(static_cast<float>(pad) / clusterer.Param().tpcGeometry.NPads(row));
    } else {
      (clusterer.nnInternals)->inputData32[write_idx] = clusterer.mISlice / 36.f;
      (clusterer.nnInternals)->inputData32[write_idx + 1] = row / 152.f;
      (clusterer.nnInternals)->inputData32[write_idx + 2] = static_cast<float>(pad) / clusterer.Param().tpcGeometry.NPads(row);
    }
  }
}

GPUd() void GPUTPCNNClusterizer::publishClustersReg1(uint glo_idx, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  CPU_ONLY(MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem));
  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
  uint full_glo_idx = glo_idx + batchStart;
  int model_output_index = glo_idx * (clusterer.nnInternals)->model_reg_1.getNumOutputNodes()[0][1];

  // LOG(info) << glo_idx << " -- " << model_output_index << " / " << (clusterer.nnInternals)->outputDataReg1.size() << " / " << (clusterer.nnInternals)->model_reg_1.getNumOutputNodes()[0][1] << " -- " << (clusterer.nnInternals)->peakPositions.size() << " -- " << (clusterer.nnInternals)->centralCharges.size();

  if ((clusterer.nnInternals)->outputDataClass[full_glo_idx] == 1) {

    ClusterAccumulator pc;

    // Publishing logic is taken from default clusterizer
    if (onlyMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect((clusterer.nnInternals)->peakPositions[glo_idx], chargeMap[(clusterer.nnInternals)->peakPositions[glo_idx]].unpack()));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        (clusterer.nnInternals)->peakPositions[glo_idx],
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }

    if ((clusterer.mPmemory->fragment).isOverlap((clusterer.nnInternals)->peakPositions[glo_idx].time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    pc.setFull((clusterer.nnInternals)->centralCharges[glo_idx] * (clusterer.nnInternals)->outputDataReg1[model_output_index + 4], static_cast<float>((clusterer.nnInternals)->peakPositions[glo_idx].pad()) + (clusterer.nnInternals)->outputDataReg1[model_output_index], (clusterer.nnInternals)->outputDataReg1[model_output_index + 2], static_cast<float>((clusterer.mPmemory->fragment).start) + static_cast<float>((clusterer.nnInternals)->peakPositions[glo_idx].time()) + (clusterer.nnInternals)->outputDataReg1[model_output_index + 1], (clusterer.nnInternals)->outputDataReg1[model_output_index + 3], 0, 0);

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative((clusterer.nnInternals)->peakPositions[glo_idx], (clusterer.nnInternals)->centralCharges[glo_idx], myCluster, clusterer.Param());
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint rowIndex = 0;
    if (clusterer.mPclusterByRow != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        (clusterer.nnInternals)->peakPositions[glo_idx].row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    CPU_ONLY(labelAcc->commit((clusterer.nnInternals)->peakPositions[glo_idx].row(), rowIndex, clusterer.mNMaxClusterPerRow));
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

GPUd() void GPUTPCNNClusterizer::publishClustersReg2(uint glo_idx, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  CPU_ONLY(MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem));
  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
  uint full_glo_idx = glo_idx + batchStart;
  int model_output_index = glo_idx * (clusterer.nnInternals)->model_reg_2.getNumOutputNodes()[0][1];

  // LOG(info) << glo_idx << " -- " << model_output_index << " / " << (clusterer.nnInternals)->outputDataReg1.size() << " / " << (clusterer.nnInternals)->model_reg_1.getNumOutputNodes()[0][1] << " -- " << (clusterer.nnInternals)->peakPositions.size() << " -- " << (clusterer.nnInternals)->centralCharges.size();

  if ((clusterer.nnInternals)->outputDataClass[full_glo_idx] > 0) {

    ClusterAccumulator pc;

    if (onlyMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect((clusterer.nnInternals)->peakPositions[glo_idx], chargeMap[(clusterer.nnInternals)->peakPositions[glo_idx]].unpack()));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        (clusterer.nnInternals)->peakPositions[glo_idx],
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }

    if ((clusterer.mPmemory->fragment).isOverlap((clusterer.nnInternals)->peakPositions[glo_idx].time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    // Cluster 1
    pc.setFull((clusterer.nnInternals)->centralCharges[glo_idx] * (clusterer.nnInternals)->outputDataReg2[model_output_index + 8], (clusterer.nnInternals)->peakPositions[glo_idx].pad() + (clusterer.nnInternals)->outputDataReg2[model_output_index], (clusterer.nnInternals)->outputDataReg2[model_output_index + 4], (clusterer.mPmemory->fragment).start + (clusterer.nnInternals)->peakPositions[glo_idx].time() + (clusterer.nnInternals)->outputDataReg2[model_output_index + 2], (clusterer.nnInternals)->outputDataReg2[model_output_index + 6], 0, 0);

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative((clusterer.nnInternals)->peakPositions[glo_idx], (clusterer.nnInternals)->centralCharges[glo_idx], myCluster, clusterer.Param());
    if (rejectCluster) {
      if ((clusterer.nnInternals)->nnClusterizerVerbosity < 2) {
        LOG(warning) << "[NN, CF] Cluster rejected!";
      }
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint rowIndex = 0;
    if (clusterer.mPclusterByRow != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        (clusterer.nnInternals)->peakPositions[glo_idx].row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    CPU_ONLY(labelAcc->commit((clusterer.nnInternals)->peakPositions[glo_idx].row(), rowIndex, clusterer.mNMaxClusterPerRow));

    // Cluster 2
    pc.setFull((clusterer.nnInternals)->centralCharges[glo_idx] * (clusterer.nnInternals)->outputDataReg2[model_output_index + 9], (clusterer.nnInternals)->peakPositions[glo_idx].pad() + (clusterer.nnInternals)->outputDataReg2[model_output_index + 1], (clusterer.nnInternals)->outputDataReg2[model_output_index + 5], (clusterer.mPmemory->fragment).start + (clusterer.nnInternals)->peakPositions[glo_idx].time() + (clusterer.nnInternals)->outputDataReg2[model_output_index + 3], (clusterer.nnInternals)->outputDataReg2[model_output_index + 7], 0, 0);

    rejectCluster = !pc.toNative((clusterer.nnInternals)->peakPositions[glo_idx], (clusterer.nnInternals)->centralCharges[glo_idx], myCluster, clusterer.Param());
    if (rejectCluster) {
      if ((clusterer.nnInternals)->nnClusterizerVerbosity < 2) {
        LOG(warning) << "[NN, CF] Cluster rejected!";
      }
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    if (clusterer.mPclusterByRow != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        (clusterer.nnInternals)->peakPositions[glo_idx].row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    // CPU_ONLY(labelAcc->commit((clusterer.nnInternals)->peakPositions[glo_idx].row(), rowIndex, clusterer.mNMaxClusterPerRow)); // -> Is this needed? How to handle MC labels for split clusters?
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}