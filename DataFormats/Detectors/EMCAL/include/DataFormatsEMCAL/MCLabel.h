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

/// \file MCLabel.h
/// \class MCLabel
/// \brief Declaration of a transient MC label class for EMCal
/// \ingroup EMCALDataFormat
/// \author anders.knospe@cern.ch

#ifndef DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_MCLABEL_H_
#define DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_MCLABEL_H_

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace emcal
{
class MCLabel : public o2::MCCompLabel
{
 private:
  double mAmplitudeFraction;

 public:
  MCLabel() = default;
  MCLabel(int trackID, int eventID, int srcID, bool fake, double afraction) : o2::MCCompLabel(trackID, eventID, srcID, fake), mAmplitudeFraction(afraction) {}
  MCLabel(bool noise, double afraction) : o2::MCCompLabel(noise), mAmplitudeFraction(afraction) {}
  void setAmplitudeFraction(double afraction) { mAmplitudeFraction = afraction; }
  double getAmplitudeFraction() const { return mAmplitudeFraction; }

  ClassDefNV(MCLabel, 1);
};
} // namespace emcal
} // namespace o2

#endif // DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_MCLABEL_H_
