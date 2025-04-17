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
#include <cstring>
#include <string>
#include <iostream>

// Executables behave this way
int callMain(int argc, char** argv, char const* pluginName);

#define STRINGIZE_NX(A) #A
#define STRINGIZE(A) STRINGIZE_NX(A)

consteval char const* pluginName()
{
  return STRINGIZE(DPL_WORKFLOW_PLUGIN_LIBRARY) ":" STRINGIZE(DPL_WORKFLOW_PLUGIN_NAME);
}

int main(int argc, char** argv)
{
  // Allow this code to lunch different plugins compared to the one
  // associated with it.
  auto pluginOption = "--workflow-plugin";
  int optionSize = strlen(pluginOption);
  std::string pluginSpec = "";
  for (int i = 0; i < argc; i++) {
    int argSize = strlen(argv[i]);
    if (argSize < optionSize) {
      continue;
    }
    if (argSize > optionSize && argv[i][optionSize] == '=') {
      pluginSpec = std::string_view(argv[i] + optionSize + 1);
      break;
    }
    if (argSize == optionSize && (strncmp(argv[i], pluginOption, optionSize) == 0) && i < argc) {
      pluginSpec = argv[i + 1];
      break;
    }
  }

  if (pluginSpec.empty()) {
    pluginSpec = pluginName();
  }

  std::cerr << "Loading plugin " << pluginSpec << std::endl;
  return callMain(argc, argv, pluginSpec.c_str());
}
