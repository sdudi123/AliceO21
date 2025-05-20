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

/// \file ShishKebabTrd1Module.cxx
/// \class ShishKebabTrd1Module
/// \brief Main class for TRD1 geometry of Shish-Kebab case.
/// \ingroup EMCALbase
/// \author Alexei Pavlinov (WSU).

#include <cassert>
#include <string>
#include <cstdio>

#include "RStringView.h"

#include "EMCALBase/Geometry.h"
#include "EMCALBase/ShishKebabTrd1Module.h"
#include "CommonConstants/MathConstants.h"

using namespace o2::emcal;
using namespace o2::constants::math;

double ShishKebabTrd1Module::sa = 0.;
double ShishKebabTrd1Module::sa2 = 0.;
double ShishKebabTrd1Module::sb = 0.;
double ShishKebabTrd1Module::sr = 0.;
double ShishKebabTrd1Module::sangle = 0.;   // around one degree
double ShishKebabTrd1Module::stanBetta = 0; //

ShishKebabTrd1Module::ShishKebabTrd1Module(double theta, Geometry* g)
  : mGeometry(g),
    mOK(),
    mTheta(theta),
    mOK1(),
    mOK2(),
    mOB(),
    mOB1(),
    mOB2(),
    mOK3X3(),
    mORB(),
    mORT()
{
  std::string_view sname = g->GetName();
  int key = 0;
  if (sname.find("v1") != std::string::npos || sname.find("V1") != std::string::npos) {
    key = 1; // EMCAL_COMPLETEV1 vs EMCAL_COMPLETEv1 (or other)
  }

  if (SetParameters()) {
    DefineFirstModule(key);
  }

  // DefineName(mTheta);
  LOG(debug4) << "o2::emcal::ShishKebabTrd1Module - first module key=" << key << ":  theta " << std::setw(1)
              << std::setprecision(4) << mTheta << " geometry " << g;
}

ShishKebabTrd1Module::ShishKebabTrd1Module(ShishKebabTrd1Module& leftNeighbor)
  : mGeometry(leftNeighbor.mGeometry),
    mOK(),
    mTheta(0.),
    mOK1(),
    mOK2(),
    mOB(),
    mOB1(),
    mOB2(),
    mOK3X3(),
    mORB(),
    mORT()
{
  //  printf("** Left Neighbor : %s **\n", leftNeighbor.GetName());
  mTheta = leftNeighbor.GetTheta() - sangle;
  Init(leftNeighbor.GetA(), leftNeighbor.GetB());
}

ShishKebabTrd1Module::ShishKebabTrd1Module(const ShishKebabTrd1Module& mod)
  : mGeometry(mod.mGeometry),
    mOK(mod.mOK),
    mA(mod.mA),
    mB(mod.mB),
    mThetaA(mod.mThetaA),
    mTheta(mod.mTheta),
    mOK1(mod.mOK1),
    mOK2(mod.mOK2),
    mOB(mod.mOB),
    mOB1(mod.mOB1),
    mOB2(mod.mOB2),
    mThetaOB1(mod.mThetaOB1),
    mThetaOB2(mod.mThetaOB2),
    mORB(mod.mORB),
    mORT(mod.mORT)
{
  for (int i = 0; i < 3; i++) {
    mOK3X3[i] = mod.mOK3X3[i];
  }
}

void ShishKebabTrd1Module::Init(double A, double B)
{
  // Define parameter module from parameters A,B from previous.
  double yl = (sb / 2) * std::sin(mTheta) + (sa / 2) * std::cos(mTheta) + sr, y = yl;
  double xl = (yl - B) / A; // y=A*x+B

  //  double xp1 = (fga/2. + fgb/2.*fgtanBetta)/(std::sin(fTheta) + fgtanBetta*std::cos(fTheta));
  //  printf(" xp1 %9.3f \n ", xp1);
  // xp1 == xp => both methods give the same results - 3-feb-05
  double alpha = PIHalf + sangle / 2;
  double xt =
    (sa + sa2) * std::tan(mTheta) * std::tan(alpha) / (4. * (1. - std::tan(mTheta) * std::tan(alpha)));
  double yt = xt / std::tan(mTheta), xp = std::sqrt(xt * xt + yt * yt);
  double x = xl + xp;
  mOK.Set(x, y);
  //  printf(" yl %9.3f | xl %9.3f | xp %9.3f \n", yl, xl, xp);

  // have to define A and B;
  double yCprev = sr + sa * std::cos(mTheta);
  double xCprev = (yCprev - B) / A;
  double xA = xCprev + sa * std::sin(mTheta), yA = sr;

  mThetaA = mTheta - sangle / 2.;
  mA = std::tan(mThetaA); // !!
  mB = yA - mA * xA;

  DefineAllStuff();
}

void ShishKebabTrd1Module::DefineAllStuff()
{
  // Define some parameters
  // DefineName(mTheta);
  // Centers of cells - 2X2 case
  double kk1 = (sa + sa2) / (2. * 4.); // kk1=kk2

  double xk1 = mOK.X() - kk1 * std::sin(mTheta);
  double yk1 = mOK.Y() + kk1 * std::cos(mTheta) - sr;
  mOK1.Set(xk1, yk1);

  double xk2 = mOK.X() + kk1 * std::sin(mTheta);
  double yk2 = mOK.Y() - kk1 * std::cos(mTheta) - sr;
  mOK2.Set(xk2, yk2);

  // Centers of cells - 3X3 case; Nov 9,2006
  mOK3X3[1].Set(mOK.X(), mOK.Y() - sr); // coincide with module center

  kk1 = ((sa + sa2) / 4. + sa / 6.) / 2.;

  xk1 = mOK.X() - kk1 * std::sin(mTheta);
  yk1 = mOK.Y() + kk1 * std::cos(mTheta) - sr;
  mOK3X3[0].Set(xk1, yk1);

  xk2 = mOK.X() + kk1 * std::sin(mTheta);
  yk2 = mOK.Y() - kk1 * std::cos(mTheta) - sr;
  mOK3X3[2].Set(xk2, yk2);

  // May 15, 2006; position of module(cells) center face
  mOB.Set(mOK.X() - sb / 2. * std::cos(mTheta), mOK.Y() - sb / 2. * std::sin(mTheta) - sr);
  mOB1.Set(mOB.X() - sa / 4. * std::sin(mTheta), mOB.Y() + sa / 4. * std::cos(mTheta));
  mOB2.Set(mOB.X() + sa / 4. * std::sin(mTheta), mOB.Y() - sa / 4. * std::cos(mTheta));
  // Jul 30, 2007 - for taking into account a position of shower maximum
  mThetaOB1 = mTheta - sangle / 4.; // ??
  mThetaOB2 = mTheta + sangle / 4.;

  // Position of right/top point of module
  // Gives the posibility to estimate SM size in z direction
  double xBottom = (sr - mB) / mA;
  double yBottom = sr;
  mORB.Set(xBottom, yBottom);

  double l = sb / std::cos(sangle / 2.); // length of lateral module side
  double xTop = xBottom + l * std::cos(std::atan(mA));
  double yTop = mA * xTop + mB;
  mORT.Set(xTop, yTop);
}

void ShishKebabTrd1Module::DefineFirstModule(const int key)
{
  // Define first module
  if (key == 0) {
    // theta in radians ; first object theta=pi/2.
    mTheta = PIHalf;
    mOK.Set(sa2 / 2., sr + sb / 2.); // position the center of module vs o

    // parameters of right line : y = A*z + B in system where zero point is IP.
    mThetaA = mTheta - sangle / 2.;
    mA = std::tan(mThetaA);
    double xA = sa / 2. + sa2 / 2.;
    double yA = sr;
    mB = yA - mA * xA;
  } else if (key == 1) {
    // theta in radians ; first object theta = 90-0.75 = 89.25 degree
    mTheta = 89.25 * Deg2Rad;
    double al1 = sangle / 2.;
    double x = 0.5 * (sa * std::cos(al1) + sb * std::sin(al1));
    double y = 0.5 * (sb + sa * std::sin(al1)) * std::cos(al1);
    mOK.Set(x, sr + y);
    // parameters of right line : y = A*z + B in system where zero point is IP.
    mThetaA = mTheta - sangle / 2.;
    mA = std::tan(mThetaA);
    double xA = sa * std::cos(al1);
    double yA = sr;
    mB = yA - mA * xA;
  } else {
    LOG(error) << "key=" << key << " : wrong case \n";
    assert(0);
  }

  DefineAllStuff();
}

bool ShishKebabTrd1Module::SetParameters()
{
  if (!mGeometry) {
    LOG(warning) << "GetParameters(): << No geometry\n";
    return kFALSE;
  }

  TString sn(mGeometry->GetName()); // 2-Feb-05
  sn.ToUpper();

  sa = static_cast<double>(mGeometry->GetEtaModuleSize());
  sb = static_cast<double>(mGeometry->GetLongModuleSize());
  sangle = static_cast<double>(mGeometry->GetTrd1Angle()) * Deg2Rad;
  stanBetta = std::tan(sangle / 2.);
  sr = static_cast<double>(mGeometry->GetIPDistance());

  sr += mGeometry->GetSteelFrontThickness();

  sa2 = static_cast<double>(mGeometry->Get2Trd1Dx2());
  // PH  PrintShish(0);
  return kTRUE;
}

//
// Service methods
//

void ShishKebabTrd1Module::PrintShish(int pri) const
{
  if (pri >= 0) {
    if (pri >= 1) {
      LOGF(info, "PrintShish() \n a %7.3f:%7.3f | b %7.2f | r %7.2f \n TRD1 angle %7.6f(%5.2f) | tanBetta %7.6f", sa, sa2,
           sb, sr, sangle, sangle * Rad2Deg, stanBetta);
      LOGF(info, " fTheta %f : %5.2f : cos(theta) %f", mTheta, GetThetaInDegree(), std::cos(mTheta));
      LOGF(info, " OK : theta %f :  phi = %f(%5.2f)", mTheta, mOK.Phi(), mOK.Phi() * Rad2Deg);
    }

    LOGF(info, " y %9.3f x %9.3f xrb %9.3f (right bottom on r=%9.3f )", mOK.X(), mOK.Y(), mORB.X(), mORB.Y());

    if (pri >= 2) {
      LOGF(info, " A %f B %f | fThetaA %7.6f(%5.2f)", mA, mB, mThetaA, mThetaA * Rad2Deg);
      LOGF(info, " fOK  : X %9.4f: Y %9.4f : eta  %5.3f", mOK.X(), mOK.Y(), GetEtaOfCenterOfModule());
      LOGF(info, " fOK1 : X %9.4f: Y %9.4f :   (local, ieta=2)", mOK1.X(), mOK1.Y());
      LOGF(info, " fOK2 : X %9.4f: Y %9.4f :   (local, ieta=1)\n", mOK2.X(), mOK2.Y());
      LOGF(info, " fOB  : X %9.4f: Y %9.4f", mOB.X(), mOB.Y());
      LOGF(info, " fOB1 : X %9.4f: Y %9.4f (local, ieta=2)", mOB1.X(), mOB1.Y());
      LOGF(info, " fOB2 : X %9.4f: Y %9.4f (local, ieta=1)", mOB2.X(), mOB2.Y());
      // 3X3
      LOGF(info, " 3X3");
      for (int ieta = 0; ieta < 3; ieta++) {
        LOGF(info, " fOK3X3[%i] : X %9.4f: Y %9.4f (local)", ieta, mOK3X3[ieta].X(), mOK3X3[ieta].Y());
      }
      //      fOK.Dump();
      GetMaxEtaOfModule();
    }
  }
}

double ShishKebabTrd1Module::GetThetaInDegree() const { return mTheta * Rad2Deg; }

double ShishKebabTrd1Module::GetEtaOfCenterOfModule() const { return -std::log(std::tan(mOK.Phi() / 2.)); }

void ShishKebabTrd1Module::GetPositionAtCenterCellLine(int ieta, double dist, TVector2& v) const
{
  // Jul 30, 2007
  double theta = 0., x = 0., y = 0.;
  if (ieta == 0) {
    v = mOB2;
    theta = mTheta;
  } else if (ieta == 1) {
    v = mOB1;
    theta = mTheta;
  } else {
    assert(0);
  }

  x = v.X() + std::cos(theta) * dist;
  y = v.Y() + std::sin(theta) * dist;
  //  printf(" GetPositionAtCenterCellLine() %s : dist %f : ieta %i : x %f %f v.X() | y %f %f v.Y() : cos %f sin %f \n",
  // GetName(), dist, ieta, v.X(),x, y,v.Y(),std::cos(theta),std::sin(theta));
  v.Set(x, y);
}

double ShishKebabTrd1Module::GetMaxEtaOfModule() const
{
  // Right bottom point of module
  double thetaBottom = std::atan2(mORB.Y(), mORB.X());
  double etaBottom = ThetaToEta(thetaBottom);

  // Right top point of module
  double thetaTop = std::atan2(mORT.Y(), mORT.X());
  double etaTop = ThetaToEta(thetaTop);

  LOG(debug) << " Right bottom point of module : eta " << std::setw(5) << std::setprecision(4) << etaBottom
             << " : theta " << std::setw(6) << std::setprecision(4) << thetaBottom << " (" << std::setw(6)
             << std::setprecision(2) << thetaBottom * Rad2Deg << " ) : x(zglob) " << std::setw(7)
             << std::setprecision(2) << mORB.X() << " y(phi) " << std::setw(5) << std::setprecision(2) << mORB.Y();
  LOG(debug) << " Right    top point of module : eta " << std::setw(5) << std::setprecision(4) << etaTop << ": theta "
             << std::setw(6) << std::setprecision(4) << thetaTop << " (" << std::setw(6) << std::setprecision(2)
             << thetaTop * Rad2Deg << ") : x(zglob) " << std::setw(7) << std::setprecision(2) << mORT.X()
             << "  y(phi) " << std::setw(5) << std::setprecision(2) << mORT.Y();
  return etaBottom > etaTop ? etaBottom : etaTop;
}
