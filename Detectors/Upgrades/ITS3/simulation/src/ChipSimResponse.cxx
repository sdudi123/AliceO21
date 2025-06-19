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

#include "ITS3Simulation/ChipSimResponse.h"
#include <vector>
#include <algorithm>

using namespace o2::its3;

ClassImp(o2::its3::ChipSimResponse);

void ChipSimResponse::initData(int tableNumber, std::string dataPath, const bool quiet)
{
  AlpideSimResponse::initData(tableNumber, dataPath, quiet);
  computeCentreFromData();
}

void ChipSimResponse::computeCentreFromData()
{
  std::vector<float> zVec, qVec;
  const int npix = o2::itsmft::AlpideRespSimMat::getNPix();

  for (int iz = 0; iz < mNBinDpt; ++iz) {
    size_t bin = iz + mNBinDpt * (0 + mNBinRow * 0);
    const auto& mat = mData[bin];
    float val = mat.getValue(npix / 2, npix / 2);
    float gz = mDptMin + iz / mStepInvDpt;
    zVec.push_back(gz);
    qVec.push_back(val);
  }

  std::vector<std::pair<float, float>> zqPairs;
  for (size_t i = 0; i < zVec.size(); ++i) {
    zqPairs.emplace_back(zVec[i], qVec[i]);
  }
  std::sort(zqPairs.begin(), zqPairs.end());
  zVec.clear();
  qVec.clear();
  for (auto& p : zqPairs) {
    zVec.push_back(p.first);
    qVec.push_back(p.second);
  }

  struct BinInfo { float z0, z1, q0, q1, dq; };
  std::vector<BinInfo> bins;
  float totQ = 0.f;
  for (size_t i = 0; i + 1 < zVec.size(); ++i) {
    float z0 = zVec[i], z1 = zVec[i + 1];
    float q0 = qVec[i], q1 = qVec[i + 1];
    float dz = z1 - z0;
    float dq = 0.5f * (q0 + q1) * dz;
    bins.push_back({z0, z1, q0, q1, dq});
    totQ += dq;
  }
  if (totQ <= 0.f) { mRespCentreDep = 0.f; return; }

  float halfQ = 0.5f * totQ;
  float cumQ = 0.f;
  for (const auto& b : bins) {
    if (cumQ + b.dq < halfQ) { cumQ += b.dq; continue; }
    float qSlope = (b.q1 - b.q0) / (b.z1 - b.z0);
    float dz = b.z1 - b.z0;
    float A = qSlope * 0.5f;
    float B = b.q0;
    float C = cumQ - halfQ;
    float disc = B * B - 4.f * A * C;
    float x;
    if (disc >= 0.f && std::abs(A) > 1.e-12f)
      x = (-B + std::sqrt(disc)) / (2.f * A);
    else
      x = (halfQ - cumQ) / b.q0;
    x = std::clamp(x, 0.f, dz);
    mRespCentreDep = b.z0 + x;
    return;
  }
  mRespCentreDep = mDptMax;
}
