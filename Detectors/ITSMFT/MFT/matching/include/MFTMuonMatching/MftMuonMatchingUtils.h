#ifndef O2_MFT_MUON_MATCHING_UTILS_H
#define O2_MFT_MUON_MATCHING_UTILS_H

/// @file   MftMuonMatchingUtils.h
/// @brief  Template-based utilities for MFT-MUON matching in ALICE O2.

#include <cmath>
#include <memory>

#include <TLorentzVector.h>
#include <TVector3.h>

#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include "TGeoManager.h"
#include "TDatabasePDG.h"

#include "Framework/Logger.h"
#include "Field/MagneticField.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/TrackFwd.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/MatchInfoFwd.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "GlobalTracking/MatchGlobalFwdParam.h"
#include "GlobalTracking/MatchGlobalFwd.h"
#include "DetectorsBase/GeometryManager.h"

namespace o2
{
namespace mft
{

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

template <typename MUON, typename MFT, typename Collision>
class MftMuonMatchingUtils
{
 private:
  MUON muontrack;
  MFT mfttrack;
  Collision collision;

  float mDX, mDY, mDPt, mDPhi, mDEta, mDInvPt, mDTanl;
  float mGlobalMuonPAtDCA, mGlobalMuonPtAtDCA, mGlobalMuonEtaAtDCA, mGlobalMuonPhiAtDCA, mGlobalMuonDCAx, mGlobalMuonDCAy;

  float mMatchingMFTX, mMatchingMFTY, mMatchingMFTEta, mMatchingMFTPhi, mMatchingMFTPt, mMatchingMFTInvPt, mMatchingMFTTanl;
  float mMatchingMuonX, mMatchingMuonY, mMatchingMuonEta, mMatchingMuonPhi, mMatchingMuonPt, mMatchingMuonInvPt, mMatchingMuonTanl;

  bool mUseMuonInfoToMFTExtrap;
  int mGlobalMuonQ;
  int mMatchingType;

  o2::field::MagneticField* fieldB;
  o2::globaltracking::MatchGlobalFwd mMatching;

  inline o2::track::TrackParCovFwd propagateMFTtoMatchingPlane()
  {

    double covArr[15]{0.0};
    SMatrix55 tmftcovs;
    std::copy(std::begin(covArr), std::end(covArr), tmftcovs.Array());

    auto muontrack_at_pv = propagateMUONtoPV();
    float total_p = muontrack_at_pv.getP();
    float mft_theta = std::numbers::pi / 2. - std::atan(mfttrack.tgl());

    float total_px = total_p * std::sin(mft_theta) * std::cos(mfttrack.phi());
    float total_py = total_p * std::sin(mft_theta) * std::sin(mfttrack.phi());
    float total_pt = std::hypot(total_px, total_py);
    float total_signed1pt = 0;

    if (mUseMuonInfoToMFTExtrap)
      total_signed1pt = muontrack.sign() / total_pt;
    else
      total_signed1pt = mfttrack.signed1Pt();

    SMatrix5 tmftpars(mfttrack.x(),
                      mfttrack.y(),
                      mfttrack.phi(),
                      mfttrack.tgl(),
                      total_signed1pt);

    o2::track::TrackParCovFwd extrap_mfttrack{mfttrack.z(),
                                              tmftpars,
                                              tmftcovs,
                                              mfttrack.chi2()};

    double propVec[3] = {0.};

    float zPlane = 0.f;
    if (mMatchingType == MCH_FIRST_CLUSTER) {
      propVec[0] = muontrack.x() - mfttrack.x();
      propVec[1] = muontrack.y() - mfttrack.y();
      propVec[2] = muontrack.z() - mfttrack.z();
      zPlane = muontrack.z();
    } else if (mMatchingType == END_OF_ABSORBER || mMatchingType == BEGINNING_OF_ABSORBER) {
      auto extrap_muontrack = propagateMUONtoMatchingPlane();
      propVec[0] = extrap_muontrack.getX() - mfttrack.x();
      propVec[1] = extrap_muontrack.getY() - mfttrack.y();
      propVec[2] = extrap_muontrack.getZ() - mfttrack.z();
      zPlane = (mMatchingType == END_OF_ABSORBER) ? -505.f : -90.f;
    } else {
      zPlane = mfttrack.z();
    }

    double centerZ[3] = {mfttrack.x() + propVec[0] / 2.,
                         mfttrack.y() + propVec[1] / 2.,
                         mfttrack.z() + propVec[2] / 2.};

    float Bz = fieldB->getBz(centerZ);
    extrap_mfttrack.propagateToZ(zPlane, Bz); // z in cm
    return extrap_mfttrack;
  }

  inline o2::dataformats::GlobalFwdTrack propagateMUONtoMatchingPlane()
  {
    float cov[15] = {
      muontrack.cXX(), muontrack.cXY(), muontrack.cYY(),
      muontrack.cPhiX(), muontrack.cPhiY(), muontrack.cPhiPhi(),
      muontrack.cTglX(), muontrack.cTglY(), muontrack.cTglPhi(),
      muontrack.cTglTgl(), muontrack.c1PtX(), muontrack.c1PtY(),
      muontrack.c1PtPhi(), muontrack.c1PtTgl(), muontrack.c1Pt21Pt2()};

    SMatrix5 tpars(muontrack.x(),
                   muontrack.y(),
                   muontrack.phi(),
                   muontrack.tgl(),
                   muontrack.signed1Pt());
    SMatrix55 tcovs(cov, cov + 15);
    double chi2 = muontrack.chi2();

    o2::track::TrackParCovFwd parcovmuontrack{muontrack.z(), tpars, tcovs, chi2};

    o2::dataformats::GlobalFwdTrack gtrack;
    gtrack.setParameters(tpars);
    gtrack.setZ(parcovmuontrack.getZ());
    gtrack.setCovariances(tcovs);

    auto mchtrack = mMatching.FwdtoMCH(gtrack);

    if (mMatchingType == MFT_LAST_CLUSTR) {
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchtrack, mfttrack.z());
    } else if (mMatchingType == END_OF_ABSORBER) {
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchtrack, -505.);
    } else if (mMatchingType == BEGINNING_OF_ABSORBER) {
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchtrack, -90.);
    } else if (mMatchingType == MCH_FIRST_CLUSTER) {
      o2::mch::TrackExtrap::extrapToVertexWithoutBranson(mchtrack, muontrack.z());
    }

    auto fwdtrack = mMatching.MCHtoFwd(mchtrack);

    o2::dataformats::GlobalFwdTrack extrap_muontrack;
    extrap_muontrack.setParameters(fwdtrack.getParameters());
    extrap_muontrack.setZ(fwdtrack.getZ());
    extrap_muontrack.setCovariances(fwdtrack.getCovariances());
    return extrap_muontrack;
  }

  inline o2::track::TrackParCovFwd propagateMFTtoDCA()
  {
    double covArr[15]{0.0};
    SMatrix55 tmftcovs(covArr, covArr + 15);

    SMatrix5 tmftpars(mfttrack.x(),
                      mfttrack.y(),
                      mfttrack.phi(),
                      mfttrack.tgl(),
                      mfttrack.signed1Pt());
    o2::track::TrackParCovFwd extrap_mfttrack{mfttrack.z(),
                                              tmftpars,
                                              tmftcovs,
                                              mfttrack.chi2()};

    double propVec[3] = {};
    propVec[0] = collision.posX() - mfttrack.x();
    propVec[1] = collision.posY() - mfttrack.y();
    propVec[2] = collision.posZ() - mfttrack.z();

    double centerZ[3] = {mfttrack.x() + propVec[0] / 2.,
                         mfttrack.y() + propVec[1] / 2.,
                         mfttrack.z() + propVec[2] / 2.};

    float Bz = fieldB->getBz(centerZ);
    extrap_mfttrack.propagateToZ(collision.posZ(), Bz); // z in cm
    return extrap_mfttrack;
  }

  inline o2::dataformats::GlobalFwdTrack propagateMUONtoPV()
  {
    float cov[15] = {
      muontrack.cXX(), muontrack.cXY(), muontrack.cYY(),
      muontrack.cPhiX(), muontrack.cPhiY(), muontrack.cPhiPhi(),
      muontrack.cTglX(), muontrack.cTglY(), muontrack.cTglPhi(),
      muontrack.cTglTgl(), muontrack.c1PtX(), muontrack.c1PtY(),
      muontrack.c1PtPhi(), muontrack.c1PtTgl(), muontrack.c1Pt21Pt2()};

    SMatrix5 tpars(muontrack.x(),
                   muontrack.y(),
                   muontrack.phi(),
                   muontrack.tgl(),
                   muontrack.signed1Pt());

    SMatrix55 tcovs(cov, cov + 15);
    double chi2 = muontrack.chi2();

    o2::track::TrackParCovFwd parcovmuontrack{muontrack.z(), tpars, tcovs, chi2};

    o2::dataformats::GlobalFwdTrack gtrack;
    gtrack.setParameters(tpars);
    gtrack.setZ(parcovmuontrack.getZ());
    gtrack.setCovariances(tcovs);

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
  enum MATCHING_TYPE { MCH_FIRST_CLUSTER,
                       MFT_LAST_CLUSTR,
                       END_OF_ABSORBER,
                       BEGINNING_OF_ABSORBER };

  MftMuonMatchingUtils(MUON const& muon,
                       MFT const& mft,
                       Collision const& coll,
                       int MType,
                       bool useMuon,
                       o2::field::MagneticField* field)
    : muontrack(muon), mfttrack(mft), collision(coll), mMatchingType(MType), mUseMuonInfoToMFTExtrap(useMuon), fieldB(field)
  {
  }

  inline void calcMatchingParams()
  {
    auto mfttrack_on_matchingP = propagateMFTtoMatchingPlane();
    auto muontrack_on_matchingP = propagateMUONtoMatchingPlane();

    float dphiRaw = mfttrack_on_matchingP.getPhi() - muontrack_on_matchingP.getPhi();
    float dphi = TVector2::Phi_mpi_pi(dphiRaw);
    float deta = mfttrack_on_matchingP.getEta() - muontrack_on_matchingP.getEta();

    mDX = mfttrack_on_matchingP.getX() - muontrack_on_matchingP.getX();
    mDY = mfttrack_on_matchingP.getY() - muontrack_on_matchingP.getY();
    mDPt = mfttrack_on_matchingP.getPt() - muontrack_on_matchingP.getPt();
    mDInvPt = 1. / mfttrack_on_matchingP.getPt() - 1. / muontrack_on_matchingP.getPt();
    mDTanl = mfttrack_on_matchingP.getTanl() - muontrack_on_matchingP.getTanl();
    mDPhi = dphi;
    mDEta = deta;

    mMatchingMFTX = mfttrack_on_matchingP.getX();
    mMatchingMFTY = mfttrack_on_matchingP.getY();
    mMatchingMFTEta = mfttrack_on_matchingP.getEta();
    mMatchingMFTPhi = mfttrack_on_matchingP.getPhi();
    mMatchingMFTPt = mfttrack_on_matchingP.getPt();
    mMatchingMFTInvPt = 1. / mMatchingMFTPt;
    mMatchingMFTTanl = mfttrack_on_matchingP.getTanl();

    mMatchingMuonX = muontrack_on_matchingP.getX();
    mMatchingMuonY = muontrack_on_matchingP.getY();
    mMatchingMuonEta = muontrack_on_matchingP.getEta();
    mMatchingMuonPhi = muontrack_on_matchingP.getPhi();
    mMatchingMuonPt = muontrack_on_matchingP.getPt();
    mMatchingMuonInvPt = 1. / mMatchingMuonPt;
    mMatchingMuonTanl = muontrack_on_matchingP.getTanl();
  }

  inline void calcGlobalMuonParams()
  {
    auto mfttrack_at_dca = propagateMFTtoDCA();
    auto muontrack_at_pv = propagateMUONtoPV();

    float momentum = muontrack_at_pv.getP();
    float theta = mfttrack_at_dca.getTheta();
    float phiTrack = mfttrack_at_dca.getPhi();
    float px = momentum * std::sin(theta) * std::cos(phiTrack);
    float py = momentum * std::sin(theta) * std::sin(phiTrack);
    float pz = momentum * std::cos(theta);

    mGlobalMuonQ = muontrack.sign() + mfttrack.sign();
    mGlobalMuonPtAtDCA = std::sqrt(px * px + py * py);
    mGlobalMuonPAtDCA = std::sqrt(px * px + py * py + pz * pz);
    mGlobalMuonEtaAtDCA = mfttrack_at_dca.getEta();
    mGlobalMuonPhiAtDCA = mfttrack_at_dca.getPhi();
    mGlobalMuonDCAx = mfttrack_at_dca.getX() - collision.posX();
    mGlobalMuonDCAy = mfttrack_at_dca.getY() - collision.posY();
  }

  inline float getDx() const { return mDX; }
  inline float getDy() const { return mDY; }
  inline float getDphi() const { return mDPhi; }
  inline float getDeta() const { return mDEta; }
  inline float getDpt() const { return mDPt; }
  inline float getDinvpt() const { return mDInvPt; }
  inline float getDtanl() const { return mDTanl; }
  inline float getMatchingMFTX() const { return mMatchingMFTX; }
  inline float getMatchingMFTY() const { return mMatchingMFTY; }
  inline float getMatchingMFTEta() const { return mMatchingMFTEta; }
  inline float getMatchingMFTPhi() const { return mMatchingMFTPhi; }
  inline float getMatchingMFTPt() const { return mMatchingMFTPt; }
  inline float getMatchingMFTInvPt() const { return mMatchingMFTInvPt; }
  inline float getMatchingMFTTanl() const { return mMatchingMFTTanl; }
  inline float getMatchingMuonX() const { return mMatchingMuonX; }
  inline float getMatchingMuonY() const { return mMatchingMuonY; }
  inline float getMatchingMuonEta() const { return mMatchingMuonEta; }
  inline float getMatchingMuonPhi() const { return mMatchingMuonPhi; }
  inline float getMatchingMuonPt() const { return mMatchingMuonPt; }
  inline float getMatchingMuonInvPt() const { return mMatchingMuonInvPt; }
  inline float getMatchingMuonTanl() const { return mMatchingMuonTanl; }
  inline float getGMPAtDCA() const { return mGlobalMuonPAtDCA; }
  inline float getGMPtAtDCA() const { return mGlobalMuonPtAtDCA; }
  inline float getGMEtaAtDCA() const { return mGlobalMuonEtaAtDCA; }
  inline float getGMPhiAtDCA() const { return mGlobalMuonPhiAtDCA; }
  inline float getGMDcaX() const { return mGlobalMuonDCAx; }
  inline float getGMDcaY() const { return mGlobalMuonDCAy; }
  inline float getGMDcaXY() const { return std::hypot(mGlobalMuonDCAx, mGlobalMuonDCAy); }
  inline int16_t getGMQ() const { return static_cast<int16_t>(mGlobalMuonQ); }
};

} // namespace mft
} // namespace o2

#endif // O2_MFT_MUON_MATCHING_UTILS_H
