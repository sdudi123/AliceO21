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

/// @file   MatchingTagAndProbe.h

#ifndef O2_MFT_MATCHING_TAG_AND_PROBE_H
#define O2_MFT_MATCHING_TAG_AND_PROBE_H

/// @file   MatchingTagAndProbe.h
/// @brief  Template-based utility for finding Tag and Probe pairs in ALICE O2.

#include <cmath>

#include "Math/SMatrix.h"
#include "Math/SVector.h"

#include "TLorentzVector.h"

#include "Framework/Logger.h"
#include "DetectorsBase/Propagator.h"
#include "MCHTracking/TrackExtrap.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "GlobalTracking/MatchGlobalFwd.h"

namespace o2
{
namespace mft
{

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

template <typename MUON, typename Collision>
class MatchingTagAndProbe
{
 private:
  inline void fillCovarianceArray(MUON const& muontrack, float cov[15]) const
  {
    cov[0] = muontrack.cXX();
    cov[1] = muontrack.cXY();
    cov[2] = muontrack.cYY();
    cov[3] = muontrack.cPhiX();
    cov[4] = muontrack.cPhiY();
    cov[5] = muontrack.cPhiPhi();
    cov[6] = muontrack.cTglX();
    cov[7] = muontrack.cTglY();
    cov[8] = muontrack.cTglPhi();
    cov[9] = muontrack.cTglTgl();
    cov[10] = muontrack.c1PtX();
    cov[11] = muontrack.c1PtY();
    cov[12] = muontrack.c1PtPhi();
    cov[13] = muontrack.c1PtTgl();
    cov[14] = muontrack.c1Pt21Pt2();
  }

  inline void setTagAndProbe()
  {
    if (muontrack1.pt() > muontrack2.pt()) {
      tagIdx = 0;
      probeIdx = 1;
    } else {
      tagIdx = 1;
      probeIdx = 0;
    }
  }

  o2::dataformats::GlobalFwdTrack muontrack_at_pv[2];
  TLorentzVector mDimuon;
  MUON muontrack1;
  MUON muontrack2;
  Collision collision;
  int tagIdx, probeIdx;
  int16_t mQ;

  const float mMu = 0.105658; // Muon mass (GeV/c^2)

  inline o2::dataformats::GlobalFwdTrack propagateMUONtoPV(MUON const& muontrack) const
  {
    const double mz = muontrack.z();
    const double mchi2 = muontrack.chi2();
    const float mx = muontrack.x();
    const float my = muontrack.y();
    const float mphi = muontrack.phi();
    const float mtgl = muontrack.tgl();
    const float m1pt = muontrack.signed1Pt();

    float cov[15];

    fillCovarianceArray(muontrack, cov);

    SMatrix5 tpars(mx, my, mphi, mtgl, m1pt);
    SMatrix55 tcovs(cov, cov + 15);

    o2::track::TrackParCovFwd parcovmuontrack{mz, tpars, tcovs, mchi2};

    o2::dataformats::GlobalFwdTrack gtrack;
    gtrack.setParameters(tpars);
    gtrack.setZ(parcovmuontrack.getZ());
    gtrack.setCovariances(tcovs);

    o2::globaltracking::MatchGlobalFwd mMatching;
    auto mchtrack = mMatching.FwdtoMCH(gtrack);

    o2::mch::TrackExtrap::extrapToVertex(mchtrack,
                                         collision.posX(),
                                         collision.posY(),
                                         collision.posZ(),
                                         collision.covXX(),
                                         collision.covYY());

    auto fwdtrack = mMatching.MCHtoFwd(mchtrack);
    o2::dataformats::GlobalFwdTrack extrap_muontrack;
    extrap_muontrack.setParameters(fwdtrack.getParameters());
    extrap_muontrack.setZ(fwdtrack.getZ());
    extrap_muontrack.setCovariances(fwdtrack.getCovariances());

    return extrap_muontrack;
  }

 public:
  MatchingTagAndProbe(const MUON& muon1, const MUON& muon2, const Collision& coll) : muontrack1(muon1), muontrack2(muon2), collision(coll), tagIdx(-1), probeIdx(-1), mQ(0)
  {
    mQ = muontrack1.sign() + muontrack2.sign();
    setTagAndProbe();
  }

  inline void calcMuonPairAtPV()
  {
    muontrack_at_pv[0] = propagateMUONtoPV(muontrack1);
    muontrack_at_pv[1] = propagateMUONtoPV(muontrack2);
    TLorentzVector vMuon1, vMuon2;
    vMuon1.SetPtEtaPhiM(muontrack_at_pv[0].getPt(),
                        muontrack_at_pv[0].getEta(),
                        muontrack_at_pv[0].getPhi(),
                        mMu);
    vMuon2.SetPtEtaPhiM(muontrack_at_pv[1].getPt(),
                        muontrack_at_pv[1].getEta(),
                        muontrack_at_pv[1].getPhi(),
                        mMu);
    mDimuon = vMuon1 + vMuon2;
  }

  inline int getTagMuonIndex() const { return tagIdx; }
  inline int getProbeMuonIndex() const { return probeIdx; }
  inline float getMass() const { return mDimuon.M(); }
  inline float getPt() const { return mDimuon.Pt(); }
  inline float getRap() const { return mDimuon.Rapidity(); }
  inline int16_t getCharge() const { return mQ; }
  inline const o2::dataformats::GlobalFwdTrack& getMuonAtPV(int idx) const
  {
    return muontrack_at_pv[idx];
  }
}; // end of class MatchingTagAndProbe

} // namespace mft
} // namespace o2

#endif // O2_MFT_MATCHING_TAG_AND_PROBE_H
