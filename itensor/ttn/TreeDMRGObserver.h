//
// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef __ITENSOR_TREEDMRGOBSERVER_H
#define __ITENSOR_TREEDMRGOBSERVER_H
#include "itensor/mps/DMRGObserver.h"
#include "binarytree.h"

namespace itensor {

  //
  // Class for monitoring DMRG calculations.
  // The measure and checkDone methods are virtual
  // so that behavior can be customized in a
  // derived class.
  //

  class TreeDMRGObserver : public DMRGObserver
  {
  public:
    
    TreeDMRGObserver(BinaryTree const& psi, 
		     Args const& args = Args::global());

    virtual ~TreeDMRGObserver() { }

    void virtual
    measure(Args const& args = Args::global());
    
    BinaryTree const& 
    psi() const { return psi_; }

  private:

    /////////////

    BinaryTree const& psi_;
    Real S; //von Neumann entropy
    bool printeigs;      //Print slowest decaying eigenvalues after every sweep
    int max_eigs;
    Real max_te;
    Spectrum last_spec_;

    /////////////

  }; // class TreeDMRGObserver

  inline TreeDMRGObserver::
  TreeDMRGObserver(BinaryTree const& psi, Args const& args) 
    : 
    DMRGObserver(MPS(psi.length()),args),psi_(psi),
	printeigs(args.getBool("PrintEigs",true)),
    max_eigs(-1),
    max_te(-1)
  { 
  }

void inline TreeDMRGObserver::
measure(Args const& args) //TODO Adpat the position of the measure depending of the order
    {
    //auto N = length(psi_);
	//auto ha = args.getInt("HalfSweep",0);
    auto b = args.getInt("AtBond",1);
    auto sw = args.getInt("Sweep",0);
    auto nsweep = args.getInt("NSweep",0);
    auto energy = args.getReal("Energy",0);
    auto silent = args.getBool("Silent",false);

    if(!silent && printeigs)
        {
        if(b == 1)
            {
            println();
            auto center_eigs = last_spec_.eigsKept();
            // Normalize eigs
            Real norm_eigs = 0;
            for(auto& p : center_eigs)
              norm_eigs += p;
			if (norm_eigs !=0) center_eigs /= norm_eigs;
            // Calculate entropy
            //Real S = 0;
            S = 0.0;
            for(auto& p : center_eigs)
                {
                if(p > 1E-13) S += p*log(p);
                }
            S *= -1;
            printfln("    vN Entropy at center bond b=%d = %.12f",1,S);
            printf(  "    Eigs at center bond b=%d: ",0);
            auto ten = decltype(center_eigs.size())(10);
            for(auto j : range(std::min(center_eigs.size(),ten)))
                {
                auto eig = center_eigs(j);
                if(eig < 1E-3) break;
                printf("%.4f ",eig);
                }
            println();
            }
        }

    max_eigs = std::max(max_eigs,last_spec_.numEigsKept());
    max_te = std::max(max_te,last_spec_.truncerr());
    if(!silent)
        {
        if(b == 1) 
            {
            if(!printeigs) println();
            auto swstr = (nsweep>0) ? format("%d/%d",sw,nsweep) 
                                    : format("%d",sw);
            println("    Largest link dim during sweep ",swstr," was ",(max_eigs > 1 ? max_eigs : 1));
            max_eigs = -1;
            println("    Largest truncation error: ",(max_te > 0 ? max_te : 0.));
            max_te = -1;
            printfln("    Energy after sweep %s is %.12f",swstr,energy);
            }
        }

    }


} //namespace itensor

#endif // __ITENSOR_DMRGOBSERVER_H
