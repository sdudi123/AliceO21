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

/// \file GPUTPCNNClusterizerKernels.cxx
/// \author Christian Sonnabend

#include "GPUTPCNNClusterizerKernels.h"
#include "GPUTPCCFClusterizer.h"
#include "GPUTPCGeometry.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#include "ML/3rdparty/GPUORTFloat16.h"

#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

#ifdef GPUCA_GPUCODE
#include "GPUTPCCFClusterizer.inc"
#endif

// Defining individual thread functions for data filling, determining the class label and running the CF clusterizer
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::runCfClusterizer>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (clustererNN.outputDataClass[glo_idx] == 0) { // default clusterizer should not be called in batched mode due to mess-up with thread indices
    return;
  }
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAcc(clusterer));
  tpc::ClusterNative* clusterOut = (withMC) ? nullptr : clusterer.mPclusterByRow;
  o2::gpu::GPUTPCCFClusterizer::GPUSharedMemory smem_new;
  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, smem_new, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::fillInputNN>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  uint write_idx = glo_idx * clustererNN.nnClusterizerElementSize; // Potential optimization: Either choose nnClusterizerBatchedMode as a power of 2 or calculate from threadId and blockId

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  ChargePos peak = clusterer.mPfilteredPeakPositions[glo_idx + batchStart];
  int row = static_cast<int>(peak.row()), pad = static_cast<int>(peak.pad()), time = static_cast<int>(peak.time()); // Explicit casting to avoid conversion errors
  float central_charge = static_cast<float>(chargeMap[peak].unpack());
  int row_offset = GPUTPCNNClusterizerKernels::rowOffset(row, clustererNN.nnClusterizerSizeInputRow);

#ifndef GPUCA_GPUCODE
  GPUCA_UNROLL(U(), U());
#endif
  for (int r = -clustererNN.nnClusterizerSizeInputRow; r <= clustererNN.nnClusterizerSizeInputRow; r++) {
    bool is_row_boundary = ((row + r) > (o2::tpc::constants::MAXGLOBALPADROW - 1)) || ((row + r) < 0);
    int pad_offset = is_row_boundary ? 0 : GPUTPCNNClusterizerKernels::padOffset(row, row + r);
    for (int p = -clustererNN.nnClusterizerSizeInputPad + pad_offset; p <= clustererNN.nnClusterizerSizeInputPad + pad_offset; p++) {
      bool is_boundary = is_row_boundary || GPUTPCNNClusterizerKernels::isBoundary(row + r + row_offset, pad + p, clustererNN.nnClusterizerSizeInputRow);
      for (int t = -clustererNN.nnClusterizerSizeInputTime; t <= clustererNN.nnClusterizerSizeInputTime; t++) {
        if (!is_boundary) {
          ChargePos tmp_pos(row + r, pad + p, time + t);
          if (r == 0 && !clustererNN.clusterFlags[2 * glo_idx] && CAMath::Abs(p) < 3 && CAMath::Abs(t) < 3 && p != 0 && t != 0) { // ordering is done for short circuit optimization
            clustererNN.clusterFlags[2 * glo_idx] = CfUtils::isPeak(isPeakMap[tmp_pos]);
            clustererNN.clusterFlags[2 * glo_idx + 1] = clustererNN.clusterFlags[2 * glo_idx];
          }
          if (dtype == 0) {
            clustererNN.inputData16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge);
          } else {
            clustererNN.inputData32[write_idx] = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge;
          }
        } else {
          // Filling boundary just to make sure that no values are left unintentionally
          if (dtype == 0) {
            clustererNN.inputData16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(clustererNN.nnClusterizerBoundaryFillValue));
          } else {
            clustererNN.inputData32[write_idx] = static_cast<float>(clustererNN.nnClusterizerBoundaryFillValue);
          }
        }
        write_idx++;
      }
    }
  }
  if (clustererNN.nnClusterizerAddIndexData) {
    if (dtype == 0) {
      clustererNN.inputData16[write_idx] = (OrtDataType::Float16_t)(clusterer.mISector / 36.f);
      clustererNN.inputData16[write_idx + 1] = (OrtDataType::Float16_t)(row / 152.f);
      clustererNN.inputData16[write_idx + 2] = (OrtDataType::Float16_t)(static_cast<float>(pad) / GPUTPCGeometry::NPads(row));
    } else {
      clustererNN.inputData32[write_idx] = clusterer.mISector / 36.f;
      clustererNN.inputData32[write_idx + 1] = row / 152.f;
      clustererNN.inputData32[write_idx + 2] = static_cast<float>(pad) / GPUTPCGeometry::NPads(row);
    }
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::determineClass1Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  processors.tpcNNClusterer[sector].outputDataClass[glo_idx + batchStart] = (int)(processors.tpcNNClusterer[sector].modelProbabilities[glo_idx] > processors.tpcNNClusterer[sector].nnClassThreshold);
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::determineClass2Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  auto& clustererNN = processors.tpcNNClusterer[sector];
  uint glo_idx = get_global_id(0);
  uint elem_iterator = glo_idx * clustererNN.nnClusterizerModelClassNumOutputNodes;
  float current_max_prob = 0.f; // If the neural network doesn't contain the softmax as a last layer, the outputs can range in [-infty, infty]
  uint class_label = 0;
  for (int pIdx = elem_iterator; pIdx < elem_iterator + clustererNN.nnClusterizerModelClassNumOutputNodes; pIdx++) {
    if (pIdx == elem_iterator) {
      current_max_prob = clustererNN.modelProbabilities[pIdx];
    } else {
      class_label = (clustererNN.modelProbabilities[pIdx] > current_max_prob ? pIdx : class_label);
    }
  }
  // uint class_label = std::distance(elem_iterator, std::max_element(elem_iterator, elem_iterator + clustererNN.nnClusterizerModelClassNumOutputNodes)); // Multiple outputs of the class network are the probabilities for each class. The highest one "wins"
  clustererNN.outputDataClass[glo_idx + batchStart] = class_label;
  if (class_label > 1) {
    clustererNN.clusterFlags[2 * glo_idx] = 1;
    clustererNN.clusterFlags[2 * glo_idx + 1] = 1;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishClass1Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  ChargePos peak = clusterer.mPfilteredPeakPositions[glo_idx + batchStart];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem);
  tpc::ClusterNative* clusterOut = (withMC) ? nullptr : clusterer.mPclusterByRow;
  uint full_glo_idx = glo_idx + batchStart;
  int model_output_index = glo_idx * clustererNN.nnClusterizerModelReg1NumOutputNodes;

  // LOG(info) << glo_idx << " -- " << model_output_index << " / " << clustererNN.outputDataReg1.size() << " / " << clustererNN.nnClusterizerModelReg1NumOutputNodes << " -- " << clusterer.peakPositions.size() << " -- " << clusterer.centralCharges.size();

  if (clustererNN.outputDataClass[full_glo_idx] == 1 || (clustererNN.nnClusterizerModelReg2NumOutputNodes == -1 && clustererNN.outputDataClass[full_glo_idx] >= 1)) {

    ClusterAccumulator pc;

    // Publishing logic is taken from default clusterizer
    if (withMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect(peak, central_charge));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        peak,
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }
    if ((clusterer.mPmemory->fragment).isOverlap(peak.time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    pc.setFull(central_charge * clustererNN.outputDataReg1[model_output_index + 4],
               static_cast<float>(peak.pad()) + clustererNN.outputDataReg1[model_output_index],
               clustererNN.outputDataReg1[model_output_index + 2],
               (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.outputDataReg1[model_output_index + 1],
               clustererNN.outputDataReg1[model_output_index + 3],
               clustererNN.clusterFlags[2 * glo_idx],
               clustererNN.clusterFlags[2 * glo_idx + 1]);

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint rowIndex = 0;
    if (clusterOut != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        peak.row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow));
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishClass2Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  ChargePos peak = clusterer.mPfilteredPeakPositions[glo_idx + batchStart];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem);
  tpc::ClusterNative* clusterOut = (withMC) ? nullptr : clusterer.mPclusterByRow;
  uint full_glo_idx = glo_idx + batchStart;
  int model_output_index = glo_idx * clustererNN.nnClusterizerModelReg2NumOutputNodes;

  if (clustererNN.outputDataClass[full_glo_idx] > 0) {

    ClusterAccumulator pc;

    if (withMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect(peak, central_charge));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        peak,
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }
    if ((clusterer.mPmemory->fragment).isOverlap(peak.time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    // Cluster 1
    pc.setFull(central_charge * clustererNN.outputDataReg2[model_output_index + 8],
               static_cast<float>(peak.pad()) + clustererNN.outputDataReg2[model_output_index],
               clustererNN.outputDataReg2[model_output_index + 4],
               (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.outputDataReg2[model_output_index + 2],
               clustererNN.outputDataReg2[model_output_index + 6],
               clustererNN.clusterFlags[2 * glo_idx],
               clustererNN.clusterFlags[2 * glo_idx + 1]);

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint rowIndex = 0;
    if (clusterOut != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        peak.row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow));

    // Cluster 2
    pc.setFull(central_charge * clustererNN.outputDataReg2[model_output_index + 9],
               static_cast<float>(peak.pad()) + clustererNN.outputDataReg2[model_output_index + 1],
               clustererNN.outputDataReg2[model_output_index + 5],
               (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.outputDataReg2[model_output_index + 3],
               clustererNN.outputDataReg2[model_output_index + 7],
               clustererNN.clusterFlags[2 * glo_idx],
               clustererNN.clusterFlags[2 * glo_idx + 1]);

    rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    if (clusterOut != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        peak.row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    // CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow)); // -> Is this needed? How to handle MC labels for split clusters?
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

// THe following arithmetic is done because the network is trained with a split between IROC and OROC boundary
GPUd() int GPUTPCNNClusterizerKernels::padOffset(int row_ref, int row_current)
{
  return (int)((GPUTPCGeometry::NPads(row_current) - GPUTPCGeometry::NPads(row_ref)) / 2);
}

GPUd() int GPUTPCNNClusterizerKernels::rowOffset(int row, int global_shift)
{
  return (row > 62 ? global_shift : 0);
}

GPUd() bool GPUTPCNNClusterizerKernels::isBoundary(int row, int pad, int global_shift)
{
  if (pad < 0 || row < 0) { // Faster short-circuit
    return true;
  } else if (row < 63) {
    return (pad >= static_cast<int>(GPUTPCGeometry::NPads(row)));
  } else if (row < (63 + global_shift)) { // to account for the gap between IROC and OROC. Charge will be set to -1 in order to signal boundary to the neural network
    return true;
  } else if (row < (o2::tpc::constants::MAXGLOBALPADROW + global_shift)) {
    return (pad >= static_cast<int>(GPUTPCGeometry::NPads(row - global_shift)));
  } else {
    return true;
  }
}
