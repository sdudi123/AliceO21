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
#ifndef O2_FRAMEWORK_DATATYPES_H_
#define O2_FRAMEWORK_DATATYPES_H_

#include "CommonConstants/LHCConstants.h"

#include <cstdint>
#include <limits>
#include <array>

namespace o2::aod::bc
{
enum BCFlags : uint8_t {
  ITSUPCMode = 0x1
};
}

namespace o2::aod::collision
{
enum CollisionFlagsRun2 : uint16_t {
  Run2VertexerTracks = 0x1,
  Run2VertexerZ = 0x2,
  Run2Vertexer3D = 0x4,
  // upper 8 bits for flags
  Run2VertexerTracksWithConstraint = 0x10,
  Run2VertexerTracksOnlyFitter = 0x20,
  Run2VertexerTracksMultiVertex = 0x40
};
} // namespace o2::aod::collision
namespace o2::aod::track
{
enum TrackTypeEnum : uint8_t {
  TrackIU = 0,      // track at point of innermost update (not propagated)
  Track = 1,        // propagated track
  StrangeTrack = 2, // track found by strangeness tracking at point of innermost update
  Run2Track = 254,
  Run2Tracklet = 255
};
enum TrackFlags : uint32_t {
  TrackTimeResIsRange = 0x1, // Gaussian or range
  PVContributor = 0x2,       // This track has contributed to the collision vertex fit
  OrphanTrack = 0x4,         // Track has no association with any collision vertex
  TrackTimeAsym = 0x8,       // track with an asymmetric time range
  // NOTE Highest 4 (29..32) bits reserved for PID hypothesis
};
enum TrackFlagsRun2Enum {
  ITSrefit = 0x1,
  FreeClsSPDTracklet = 0x1, // for SPD tracklets, tracklet from cluster not used in tracking
  TPCrefit = 0x2,
  GoldenChi2 = 0x4,
  TPCout = 0x8
  // NOTE Highest 4 (29..32) bits reserved for PID hypothesis
};
enum DetectorMapEnum : uint8_t {
  ITS = 0x1,
  TPC = 0x2,
  TRD = 0x4,
  TOF = 0x8
};
enum TRDTrackPattern : uint8_t {
  Layer0 = 0x1,
  Layer1 = 0x2,
  Layer2 = 0x4,
  Layer3 = 0x8,
  Layer4 = 0x10,
  Layer5 = 0x20,
  HasNeighbor = 0x40,
  HasCrossing = 0x80,
};
namespace extensions
{
struct TPCTimeErrEncoding {
  // TPC delta forward & backward packing
  union TPCDeltaTime {
    struct {
      uint16_t timeForward;
      uint16_t timeBackward;
    } __attribute__((packed)) deltas;
    float timeErr;
  } encoding;
  static_assert(sizeof(float) == 2 * sizeof(uint16_t));

  float getTimeErr() const
  {
    return encoding.timeErr;
  }

  // Use all 16 bits of uint16_t to encode delta scale with max precision
  // e.g., TPCTrack::mDeltaFwd * timeScaler
  // max range for the time deltas is 0 - <512 (1<<9) TPC time bins
  static constexpr float timeScaler{(1 << 16) / (1 << 9)};
  // bogus value to max incorrect usae immedately obvious
  static constexpr float invalidValue{std::numeric_limits<float>::min()};
  // convert TPC time bins to ns
  static constexpr float TPCBinNS = 8 * o2::constants::lhc::LHCBunchSpacingNS;

  void setDeltaTFwd(float fwd)
  {
    encoding.deltas.timeForward = static_cast<uint16_t>(fwd * timeScaler);
  }
  void setDeltaTBwd(float bwd)
  {
    encoding.deltas.timeBackward = static_cast<uint16_t>(bwd * timeScaler);
  }

  float getDeltaTFwd() const
  {
    return static_cast<float>(encoding.deltas.timeForward) / timeScaler * TPCBinNS;
  }
  float getDeltaTBwd() const
  {
    return static_cast<float>(encoding.deltas.timeBackward) / timeScaler * TPCBinNS;
  }
};
} // namespace extensions

// Reference radius for extrapolated tracks
constexpr float trackQARefRadius{50.f};
constexpr float trackQAScaleBins{5.f};
// Fit parameters for scale dY, dZ, dSnp, dTgl, dQ2Pt
constexpr std::array<float, 5> trackQAScaleContP0{0.257192, 0.0775375, 0.00424283, 0.00107201, 0.0335447};
constexpr std::array<float, 5> trackQAScaleContP1{0.189371, 0.409071, 0.00694444, 0.00720038, 0.0806902};
constexpr std::array<float, 5> trackQAScaleGloP0{0.130985, 0.0775375, 0.00194703, 0.000405458, 0.0160007};
constexpr std::array<float, 5> trackQAScaleGloP1{0.183731, 0.409071, 0.00621802, 0.00624881, 0.0418957};
constexpr std::array<float, 2> trackQAScaledTOF{1.1, 0.33};
} // namespace o2::aod::track

namespace o2::aod::mctracklabel
{
// ! Bit mask to indicate detector mismatches (bit ON means mismatch). Bit 0-6: mismatch at ITS layer. Bit 7-9: # of TPC mismatches in the ranges 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, >64. Bit 10: TRD, bit 11: TOF, bit 15: indicates negative label
enum McMaskEnum : uint16_t {
  MismatchInITS0 = 0x1,   // BIT(0) Mismatch in the layer 0 of ITS
  MismatchInITS1 = 0x2,   // BIT(1) Mismatch in the layer 1 of ITS
  MismatchInITS2 = 0x4,   // BIT(2) Mismatch in the layer 2 of ITS
  MismatchInITS3 = 0x8,   // BIT(3) Mismatch in the layer 3 of ITS
  MismatchInITS4 = 0x10,  // BIT(4) Mismatch in the layer 4 of ITS
  MismatchInITS5 = 0x20,  // BIT(5) Mismatch in the layer 5 of ITS
  MismatchInITS6 = 0x40,  // BIT(6) Mismatch in the layer 6 of ITS
  MismatchInTPC0 = 0x80,  // BIT(7) Mismatch in the 0 of TPC
  MismatchInTPC1 = 0x100, // BIT(8) Mismatch in the 1 of TPC
  MismatchInTPC2 = 0x200, // BIT(9) Mismatch in the 2 of TPC
  MismatchInTRD = 0x400,  // BIT(10) Mismatch in the TRD
  MismatchInTOF = 0x800,  // BIT(11) Mismatch in the TOF
  Noise = 0x1000,         // BIT(12)
  Fake = 0x2000,          // BIT(13)
  // MatchTPC0 = 0x4000, // BIT(14)
  NegativeLabel = 0x8000 // BIT(15) Negative label
};

} // namespace o2::aod::mctracklabel

namespace o2::aod::fwdtrack
{
enum ForwardTrackTypeEnum : uint8_t {
  GlobalMuonTrack = 0,       // MFT-MCH-MID
  GlobalMuonTrackOtherMatch, // MFT-MCH-MID (MCH-MID used another time)
  GlobalForwardTrack,        // MFT-MCH
  MuonStandaloneTrack,       // MCH-MID
  MCHStandaloneTrack         // MCH
};
} // namespace o2::aod::fwdtrack

namespace o2::aod::mcparticle::enums
{
enum MCParticleFlags : uint8_t {
  ProducedByTransport = 0x1,
  FromBackgroundEvent = 0x2,          // Particle from background event (may have been used several times)
  PhysicalPrimary = 0x4,              // Particle is a physical primary according to ALICE definition
  FromOutOfBunchPileUpCollision = 0x8 // Particle from out-of-bunch pile up collision (currently Run 2 only)
};
} // namespace o2::aod::mcparticle::enums

namespace o2::aod::run2
{
enum Run2EventSelectionCut {
  kINELgtZERO = 0,
  kPileupInMultBins,
  kConsistencySPDandTrackVertices,
  kTrackletsVsClusters,
  kNonZeroNContribs,
  kIncompleteDAQ,
  kPileUpMV,
  kTPCPileUp,
  kTimeRangeCut,
  kEMCALEDCut,
  kAliEventCutsAccepted,
  kIsPileupFromSPD,
  kIsV0PFPileup,
  kIsTPCHVdip,
  kIsTPCLaserWarmUp,
  kTRDHCO, // Offline TRD cosmic trigger decision
  kTRDHJT, // Offline TRD jet trigger decision
  kTRDHSE, // Offline TRD single electron trigger decision
  kTRDHQU, // Offline TRD quarkonium trigger decision
  kTRDHEE  // Offline TRD single-electron-in-EMCAL-acceptance trigger decision
};
} // namespace o2::aod::run2

#endif // O2_FRAMEWORK_DATATYPES_H_
