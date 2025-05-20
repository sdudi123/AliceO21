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

/// \file ShishKebabTrd1Module.h
/// \class ShishKebabTrd1Module
/// \brief Main class for TRD1 geometry of Shish-Kebab case.
/// \ingroup EMCALbase
/// \author Alexei Pavlinov (WSU).
///
/// Sep 20004 - Nov 2006; Apr 2010
/// See web page with description of Shish-Kebab geometries:
/// http://pdsfweb01.nersc.gov/~pavlinov/ALICE/SHISHKEBAB/RES/shishkebabALICE.html
/// Nov 9,2006 - added case of 3X3
///
///

#ifndef DETECTORS_EMCAL_BASE_INCLUDE_EMCALBASE_SHISHKEBABTRD1MODULE_H_
#define DETECTORS_EMCAL_BASE_INCLUDE_EMCALBASE_SHISHKEBABTRD1MODULE_H_

#include <fairlogger/Logger.h>
#include <iomanip>
#include <memory>

#include <TMath.h>
#include <TNamed.h>
#include <TVector2.h>

namespace o2
{
namespace emcal
{
class ShishKebabTrd1Module
{
 public:
  /// \brief Constructor.
  explicit ShishKebabTrd1Module(double theta = 0.0, Geometry* g = nullptr);

  /// \brief Constructor.
  ShishKebabTrd1Module(ShishKebabTrd1Module& leftNeighbor);

  /// \brief Init (add explanation)
  void Init(double A, double B);

  /// \brief Define more things (add explanation)
  void DefineAllStuff();

  /// \brief Copy Constructor.
  ShishKebabTrd1Module(const ShishKebabTrd1Module& mod);

  ShishKebabTrd1Module& operator=(const ShishKebabTrd1Module& /*rvalue*/)
  {
    LOG(fatal) << "operator = not implemented";
    return *this;
  }

  ~ShishKebabTrd1Module() = default;

  /// \brief Recover module parameters stored in geometry
  bool SetParameters();

  ///
  /// This is what we have in produced SM. (add explanation)
  ///    Oct 23-25, 2010
  ///  key=0 - zero tilt of first module;
  ///  key=1 - angle=fgangle/2 = 0.75 degree.
  ///
  void DefineFirstModule(const int key = 0); // key=0-zero tilt of first module

  double GetTheta() const { return mTheta; }
  const TVector2& GetCenterOfModule() const { return mOK; }
  double GetPosX() const { return mOK.Y(); }
  double GetPosZ() const { return mOK.X(); }
  double GetPosXfromR() const { return mOK.Y() - sr; }
  double GetA() const { return mA; }
  double GetB() const { return mB; }
  double GetRadius() const { return sr; }
  TVector2 GetORB() const { return mORB; }
  TVector2 GetORT() const { return mORT; }

  //  Additional offline stuff
  //  ieta=0 or 1 - Jun 02, 2006
  const TVector2& GetCenterOfCellInLocalCoordinateofSM(int ieta) const
  {
    if (ieta <= 0) {
      return mOK2;
    } else {
      return mOK1;
    }
  }

  void GetCenterOfCellInLocalCoordinateofSM(int ieta, double& xr, double& zr) const
  {
    if (ieta <= 0) {
      xr = mOK2.Y();
      zr = mOK2.X();
    } else {
      xr = mOK1.Y();
      zr = mOK1.X();
    }
    LOG(debug2) << " ieta " << std::setw(2) << std::setprecision(2) << ieta << " xr " << std::setw(8)
                << std::setprecision(4) << xr << " zr " << std::setw(8) << std::setprecision(4) << zr;
  }

  void GetCenterOfCellInLocalCoordinateofSM3X3(int ieta, double& xr, double& zr) const
  { // 3X3 case - Nov 9,2006
    if (ieta < 0) {
      ieta = 0; // ieta = ieta<0? ieta=0 : ieta; // check index
    }
    if (ieta > 2) {
      ieta = 2; // ieta = ieta>2? ieta=2 : ieta;
    }
    xr = mOK3X3[2 - ieta].Y();
    zr = mOK3X3[2 - ieta].X();
  }

  void GetCenterOfCellInLocalCoordinateofSM1X1(double& xr, double& zr) const
  { // 1X1 case - Nov 27,2006 // Center of cell is center of module
    xr = mOK.Y() - sr;
    zr = mOK.X();
  }

  // 15-may-06
  const TVector2& GetCenterOfModuleFace() const { return mOB; }
  const TVector2& GetCenterOfModuleFace(int ieta) const
  {
    if (ieta <= 0) {
      return mOB2;
    } else {
      return mOB1;
    }
  }

  // Jul 30, 2007
  void GetPositionAtCenterCellLine(int ieta, double dist, TVector2& v) const;

  //
  double GetTanBetta() const { return stanBetta; }
  double Getb() const { return sb; }

  // service methods
  void PrintShish(int pri = 1) const; // *MENU*
  double GetThetaInDegree() const;
  double GetEtaOfCenterOfModule() const;
  double GetMaxEtaOfModule() const;
  static double ThetaToEta(double theta) { return -std::log(std::tan(theta / 2.)); }

 protected:
  // geometry info
  Geometry* mGeometry;     //!<! pointer to geometry info
  static double sa;        ///<  2*dx1=2*dy1
  static double sa2;       ///<  2*dx2
  static double sb;        ///<  2*dz1
  static double sangle;    ///<  in rad (1.5 degree)
  static double stanBetta; ///<  tan(fgangle/2.)
  static double sr;        ///<  radius to IP

  TVector2 mOK;       ///< position the module center in ALICE system; x->y; z->x;
  double mA{0.};      ///< parameters of right line : y = A*z + B
  double mB{0.};      ///< system where zero point is IP.
  double mThetaA{0.}; ///< angle coresponding fA - for convinience
  double mTheta;      ///< theta angle of perpendicular to SK module

  // position of towers(cells) with differents ieta (1 or 2) in local coordinate of SM
  // Nov 04,2004; Feb 19,2006
  TVector2 mOK1; ///< ieta=1
  TVector2 mOK2; ///< ieta=0

  // May 13, 2006; local position of module (cells) center face
  TVector2 mOB;  ///< module
  TVector2 mOB1; ///< ieta=1
  TVector2 mOB2; ///< ieta=0

  // Jul 30, 2007
  double mThetaOB1{0.}; ///< theta of cell center line (go through OB1)
  double mThetaOB2{0.}; ///< theta of cell center line (go through OB2)

  // 3X3 case - Nov 9,2006
  TVector2 mOK3X3[3];

  // Apr 14, 2010 - checking of geometry
  TVector2 mORB; ///< position of right/bottom point of module
  TVector2 mORT; ///< position of right/top    point of module
};
} // namespace emcal
} // namespace o2
#endif // DETECTORS_EMCAL_BASE_INCLUDE_EMCALBASE_SHISHKEBABTRD1MODULE_H_
