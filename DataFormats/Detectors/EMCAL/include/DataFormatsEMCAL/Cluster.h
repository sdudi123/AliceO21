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

/// \file Cluster.h
/// \class Cluster
/// \brief EMCAL Cluster
/// \ingroup EMCALDataFormat
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
///

#ifndef DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_CLUSTER_H_
#define DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_CLUSTER_H_

#include <array>
#include <iosfwd>
#include <string>
#include <vector>
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace emcal
{
class Cluster : public o2::dataformats::TimeStamp<Float16_t>
{
  using CellIndexRange = o2::dataformats::RangeRefComp<8>;

 public:
  Cluster() = default;
  Cluster(float time, int firstcell, int ncells);
  ~Cluster() noexcept = default;

  int getNCells() const { return mCellIndices.getEntries(); }
  int getCellIndexFirst() const { return mCellIndices.getFirstEntry(); }
  CellIndexRange getCellIndexRange() const { return mCellIndices; }

  void setCellIndices(int firstcell, int ncells)
  {
    mCellIndices.setFirstEntry(firstcell);
    mCellIndices.setEntries(ncells);
  }
  void setCellIndexFirst(int firstcell) { mCellIndices.setFirstEntry(firstcell); }
  void setNCells(int ncells) { mCellIndices.setEntries(ncells); }

  void PrintStream(std::ostream& stream) const;

 private:
  CellIndexRange mCellIndices; ///< Cells contributing to a cluser
  ClassDefNV(Cluster, 1);
};

std::ostream& operator<<(std::ostream& stream, const o2::emcal::Cluster& cluster);

} // namespace emcal

} // namespace o2

#endif // DATAFORMATS_DETECTORS_EMCAL_INCLUDE_DATAFORMATSEMCAL_CLUSTER_H_
