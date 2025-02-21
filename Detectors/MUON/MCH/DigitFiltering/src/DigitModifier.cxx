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

#include "MCHDigitFiltering/DigitModifier.h"

#include "DataFormatsMCH/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include <fmt/format.h>
#include <functional>
#include <array>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <limits>

namespace o2::mch
{
DigitModifier createDigitModifier(int runNumber,
                                  bool updateST1,
                                  bool updateST2)
{
  return {};
}

} // namespace o2::mch
