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

/// \author S. Wenzel - Mai 2018

#ifndef ALICEO2_GENERATORFACTORY_H_
#define ALICEO2_GENERATORFACTORY_H_

#include "FairGenerator.h"
#include "FairBoxGenerator.h"
#include <Generators/GeneratorFromFile.h>
#include <Generators/GeneratorTParticle.h>
#ifdef GENERATORS_WITH_HEPMC3
#include <Generators/GeneratorHepMC.h>
#endif
#if defined(GENERATORS_WITH_PYTHIA8) && defined(GENERATORS_WITH_HEPMC3)
#include <Generators/GeneratorHybrid.h>
#endif
#ifdef GENERATORS_WITH_PYTHIA8
#include <Generators/GeneratorPythia8.h>
#endif

class FairPrimaryGenerator;
namespace o2
{
namespace conf
{
class SimConfig;
}
} // namespace o2

namespace o2
{
namespace eventgen
{

// reusable helper class
// main purpose is to init a FairPrimGen given some (Sim)Config
struct GeneratorFactory {
  static void setPrimaryGenerator(o2::conf::SimConfig const&, FairPrimaryGenerator*);
  // Make destructor to delete all the pointers
  ~GeneratorFactory()
  {
    cleanup();
  }
  static void cleanup()
  {
    for (auto& gen : mBoxGenPtr) {
      delete gen;
    }
    delete mPythia8GenPtr;
    delete mHybridGenPtr;
    delete mHepMCGenPtr;
    delete mExtGenPtr;
    delete mFileGenPtr;
    delete mO2KineGenPtr;
    delete mTParticleGenPtr;
  }
  static std::vector<FairBoxGenerator*> mBoxGenPtr;
  static o2::eventgen::GeneratorPythia8* mPythia8GenPtr;
  static o2::eventgen::GeneratorHybrid* mHybridGenPtr;
  static o2::eventgen::GeneratorHepMC* mHepMCGenPtr;
  static FairGenerator* mExtGenPtr;
  static o2::eventgen::GeneratorFromFile* mFileGenPtr;
  static o2::eventgen::GeneratorFromO2Kine* mO2KineGenPtr;
  static o2::eventgen::GeneratorTParticle* mTParticleGenPtr;
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_GENERATORFACTORY_H_
