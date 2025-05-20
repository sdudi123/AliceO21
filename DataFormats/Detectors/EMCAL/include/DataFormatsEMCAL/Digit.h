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

/// \file   Digit.h
/// \author anders.knospe@cern.ch
/// \brief  Definition of EMCal Digit class

#ifndef DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_DIGIT_H_
#define DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_DIGIT_H_

#include <iosfwd>
#include <cmath>
#include "Rtypes.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsEMCAL/Constants.h"

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{

namespace emcal
{
using DigitBase = o2::dataformats::TimeStamp<double>;

/// \class Digit
/// \brief EMCAL digit implementation
/// \ingroup EMCALDataFormat
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(int16_t tower, double amplitudeGeV, double time);
  Digit(int16_t tower, uint16_t noiseLG, uint16_t noiseHG, double time);
  ~Digit() = default; // override

  bool operator<(const Digit& other) const { return getTimeStamp() < other.getTimeStamp(); }
  bool operator>(const Digit& other) const { return getTimeStamp() > other.getTimeStamp(); }
  bool operator==(const Digit& other) const { return getTimeStamp() == other.getTimeStamp(); }

  bool canAdd(const Digit other)
  {
    return (mTower == other.getTower() && std::abs(getTimeStamp() - other.getTimeStamp()) < constants::EMCAL_TIMESAMPLE);
  }

  Digit& operator+=(const Digit& other);              // Adds amplitude of other digits to this digit.
  friend Digit operator+(Digit lhs, const Digit& rhs) // Adds amplitudes of two digits.
  {
    lhs += rhs;
    return lhs;
  }

  void setTower(int16_t tower) { mTower = tower; }
  int16_t getTower() const { return mTower; }

  void setAmplitude(double amplitude) { mAmplitudeGeV = amplitude; }
  double getAmplitude() const;

  void setEnergy(double energy) { mAmplitudeGeV = energy; }
  double getEnergy() const { return mAmplitudeGeV; }

  void setAmplitudeADC(int16_t amplitude, ChannelType_t ctype = ChannelType_t::HIGH_GAIN);
  int getAmplitudeADC(ChannelType_t ctype) const;
  int getAmplitudeADC() const { return getAmplitudeADC(getType()); }

  void setType(ChannelType_t ctype) {}
  ChannelType_t getType() const;

  void setHighGain() {}
  bool getHighGain() const { return (getType() == ChannelType_t::HIGH_GAIN); }

  void setLowGain() {}
  bool getLowGain() const { return (getType() == ChannelType_t::LOW_GAIN); }

  void setTRU() { mIsTRU = true; }
  bool getTRU() const { return mIsTRU; }

  void setLEDMon() {}
  bool getLEDMon() const { return false; }

  void PrintStream(std::ostream& stream) const;

  void setNoiseLG(uint16_t noise) { mNoiseLG = noise; }
  uint16_t getNoiseLG() const { return mNoiseLG; }

  void setNoiseHG(uint16_t noise) { mNoiseHG = noise; }
  uint16_t getNoiseHG() const { return mNoiseHG; }

  void setNoiseTRU(uint16_t noise) { mNoiseHG = noise; }
  uint16_t getNoiseTRU() const { return mNoiseHG; }

 private:
  friend class boost::serialization::access;

  double mAmplitudeGeV = 0.; ///< Amplitude (GeV)
  int16_t mTower = -1;       ///< Tower index (absolute cell ID)
  bool mIsTRU = false;       ///< TRU flag
  uint16_t mNoiseLG = 0;     ///< Noise of the low gain digits
  uint16_t mNoiseHG = 0;     ///< Noise of the high gain digits or TRU digits (can never be at the same time)

  ClassDefNV(Digit, 3);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace emcal
} // namespace o2
#endif // DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_DIGIT_H_
