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

///
/// \file    Location.cxx
/// \author  Julian Myrcha
///

#include "EventVisualisationDataConverter/Location.h"
#include <fairlogger/Logger.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netdb.h>

using namespace std;
namespace o2::event_visualisation
{
void Location::open()
{
  if (this->mToFile) {
    this->mOut = new std::ofstream(mFileName, std::ios::out | std::ios::binary);
  }
  if (this->mToSocket) {
    // resolve host name
    sockaddr_in serverAddress; // NOLINT(*-pro-type-member-init)
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(this->mPort); // Port number

    // ask once
    static auto server = gethostbyname(this->mHostName.c_str());
    if (server == nullptr) {
      fprintf(stderr, "ERROR, no such host\n");
      return;
    };

    bcopy((char*)server->h_addr,
          (char*)&serverAddress.sin_addr.s_addr,
          server->h_length);

    // Connect to the server
    this->mClientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (this->mClientSocket == -1) {
      LOGF(info, "Error creating socket");
      return;
    }

    if (connect(this->mClientSocket, (struct sockaddr*)&serverAddress,
                sizeof(serverAddress)) == -1) {
      LOGF(info, "Error connecting to %s:%d", this->mHostName.c_str(), this->mPort);
      ::close(this->mClientSocket);
      this->mClientSocket = -1;
      return;
    }
    try {
      char buf[256] = "SEND:";
      strncpy(buf + 6, this->mFileName.c_str(), sizeof(buf) - 7);
      strncpy(buf + sizeof(buf) - 6, "ALICE", 6);
      auto real = send(this->mClientSocket, buf, sizeof(buf), 0);
      if (real != sizeof(buf)) {
        throw real;
      }
    } catch (...) {
      ::close(this->mClientSocket);
      this->mClientSocket = -1;
      LOGF(info, "Error sending file name to %s:%d", this->mHostName.c_str(), this->mPort);
    }
  }
}

void Location::close()
{
  if (this->mToFile && this->mOut) {
    this->mOut->close();
    delete this->mOut;
    this->mOut = nullptr;
  }
  if (this->mToSocket && this->mClientSocket != -1) {
    ::close(this->mClientSocket);
    this->mClientSocket = -1;
  }
}

void Location::write(char* buf, std::streamsize size)
{
  if (size == 0) {
    return;
  }
  if (this->mToFile && this->mOut) {
    this->mOut->write(buf, size);
  }
  if (this->mToSocket && this->mClientSocket != -1) {
    try {
      auto real = send(this->mClientSocket, buf, size, 0);
      if (real != size) {
        throw real;
      }
    } catch (...) {
      ::close(this->mClientSocket);
      this->mClientSocket = -1;
      LOGF(info, "Error sending data to %s:%d", this->mHostName.c_str(), this->mPort);
    }
  }
}

} // namespace o2::event_visualisation
