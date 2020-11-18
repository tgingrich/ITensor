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
#ifndef __ITENSOR_TREE_DMRG_H
#define __ITENSOR_TREE_DMRG_H

#include "itensor/iterativesolvers.h"
#include "itensor/mps/localmposet.h"
#include "itensor/mps/localmpo_mps.h"
#include "itensor/ttn/localmpo_binarytree.h"
#include "itensor/mps/sweeps.h"
#include "itensor/util/cputime.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/ttn/TreeDMRGObserver.h"


namespace itensor {

	long subspace_expansion(BinaryTree & psi,LocalMPO_BT & PH,int b1,int b2, Real alpha);

  template<class LocalOpT>
  Real
  TreeDMRGWorker(BinaryTree & psi,
		 LocalOpT & PH,
		 Sweeps const& sweeps,
		 Args const& args = Args::global());

  template<class LocalOpT>
  Real
  TreeDMRGWorker(BinaryTree & psi,
		 LocalOpT & PH,
		 Sweeps const& sweeps,
		 TreeDMRGObserver & obs,
		 Args args = Args::global());

  //
  // Available DMRG methods:
  //

  //
  //DMRG with an MPO
  //
  Real inline
  tree_dmrg(BinaryTree & psi,
	    MPO const& H,
	    Sweeps const& sweeps,
	    Args const& args = Args::global())
  {
    LocalMPO_BT PH(H,args);
    Real energy = TreeDMRGWorker(psi,PH,sweeps,args);
    return energy;
  }

  //
  //DMRG with an MPO
  //Version that takes a starting guess BinaryTree
  //and returns the optimized BinaryTree
  //
  std::tuple<Real,BinaryTree> inline
  tree_dmrg(MPO const& H,
	    BinaryTree const& psi0,
	    Sweeps const& sweeps,
	    Args const& args = Args::global())
  {
    auto psi = psi0;
    auto energy = tree_dmrg(psi,H,sweeps,args);
    return std::tuple<Real,BinaryTree>(energy,psi);
  }

  //
  //DMRG with an MPO and custom TreeDMRGObserver
  //
  Real inline
  tree_dmrg(BinaryTree& psi,
	    MPO const& H,
	    Sweeps const& sweeps,
	    TreeDMRGObserver & obs,
	    Args const& args = Args::global())
  {
    LocalMPO_BT PH(H,args);
    Real energy = TreeDMRGWorker(psi,PH,sweeps,obs,args);
    return energy;
  }

  //
  //DMRG with an MPO and custom TreeDMRGObserver
  //Version that takes a starting guess BinaryTree
  //and returns the optimized BinaryTree
  //
  std::tuple<Real,BinaryTree> inline
  tree_dmrg(MPO const& H,
	    BinaryTree const& psi0,
	    Sweeps const& sweeps,
	    TreeDMRGObserver & obs,
	    Args const& args = Args::global())
  {
    auto psi = psi0;
    auto energy = tree_dmrg(psi,H,sweeps,obs,args);
    return std::tuple<Real,BinaryTree>(energy,psi);
  }

  //
  // TreeDMRGWorker
  //

  template<class LocalOpT>
  Real
  TreeDMRGWorker(BinaryTree & psi,
		 LocalOpT & PH,
		 Sweeps const& sweeps,
		 Args const& args)
  {
    TreeDMRGObserver obs(psi,args);
    Real energy = TreeDMRGWorker(psi,PH,sweeps,obs,args);
    return energy;
  }

  template<class LocalOpT>
  Real
  TreeDMRGWorker(BinaryTree & psi,
		 LocalOpT & PH,
		 Sweeps const& sweeps,
		 TreeDMRGObserver & obs,
		 Args args)
{
    if( args.defined("WriteM") )
      {
  if( args.defined("WriteDim") )
    {
      Global::warnDeprecated("Args WirteM and WriteDim are both defined. WriteM is deprecated in favor of WriteDim, WriteDim will be used.");
    }
  else
    {
      Global::warnDeprecated("Arg WriteM is deprecated in favor of WriteDim.");
      args.add("WriteDim",args.getInt("WriteM"));
    }
      }

    // Truncate blocks of degenerate singular values (or not)
    args.add("RespectDegenerate",args.getBool("RespectDegenerate",true));

    const bool silent = args.getBool("Silent",false);
    if(silent)
      {
        args.add("Quiet",true);
        args.add("PrintEigs",false);
        args.add("NoMeasure",true);
        args.add("DebugLevel",0);
      }
    const bool quiet = args.getBool("Quiet",false);
    const int debug_level = args.getInt("DebugLevel",(quiet ? 0 : 1));

    Real energy = NAN;

    const int numCenter = args.getInt("NumCenter",2);
  //println("nc ",numCenter);
    psi.setOrder(args); // Choose sweep order

    // psi.position(psi.startPoint(args));

    const bool subspace_exp=args.getBool("SubspaceExpansion",false);
    Real alpha = 0.0;
    args.add("DebugLevel",debug_level);
    args.add("DoNormalize",true);
    args.add("UseSVD",true);
    
    for(int sw = 1; sw <= sweeps.nsweep(); ++sw)
      {
        cpu_time sw_time;
        args.add("Sweep",sw);
        args.add("NSweep",sweeps.nsweep());
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("MinDim",sweeps.mindim(sw));
        args.add("MaxDim",sweeps.maxdim(sw));
        args.add("Noise",sweeps.noise(sw));
        args.add("MaxIter",sweeps.niter(sw));

        // args.add("WhichEig","LargestReal");

        if(subspace_exp)
        {
          alpha = sweeps.alpha(sw);
        }

        if(!PH.doWrite()
           && args.defined("WriteDim")
           && sweeps.maxdim(sw) >= args.getInt("WriteDim"))
    {
            if(!quiet)
        {
                println("\nTurning on write to disk, write_dir = ",
                        args.getString("WriteDir","./"));
        }

            psi.doWrite(true);
            PH.doWrite(true,args);
    }
  ITensor phi;
  Spectrum spec;


        for(int b = psi.startPoint(args), ha = 1; ha <= 2; sweepnext(b,ha,psi,args)) // Do one sweep go and return
    {
            if(!quiet)
        {
                printfln("Sweep=%d, HS=%d, Bond=%d/%d",sw,ha,b,psi.size()-1);
        }

      TIMER_START(1);
      psi.position(b,args); //Orthogonalize with respect to b

        PH.position(b,psi); // Compute the local environnement
      TIMER_STOP(1);

      TIMER_START(2);
      // The local vector to update
      if (numCenter == 2) phi = psi(b)*psi(psi.parent(b));
            else if(numCenter == 1) phi = psi(b);
      TIMER_STOP(2);

      // PrintData(psi);
      // PrintData(phi);
      // if (b == 1 || b == 2) {
      //   PrintData(PH.lop().L() * PH.lop().Op1() * PH.lop().R());
      // } else {
      //   PrintData(PH.lop().L() * PH.lop().Op1() * PH.lop().Op2() * PH.lop().R());
      // }

      TIMER_START(3);
            energy = arnoldi(PH,phi,args).real();
            phi.takeReal();
      TIMER_STOP(3);

      TIMER_START(4);
      //Restore tensor network form
            if (numCenter == 2) 
        {
      spec = psi.svdBond(ha==1?b:psi.parent(b),phi,ha==1?psi.parent(b):b,PH,args);//That change to make direction depend of sweep direction
      PH.haveBeenUpdated(b);
      PH.haveBeenUpdated(psi.parent(b)); // To known that we need to update the environement tensor
        }
      else if(numCenter == 1)
        {
      psi.ref(b) = phi;
      PH.haveBeenUpdated(b);
        }

      // PrintData(phi);
      // PrintData(psi);
      // PrintData(psi(b) * psi(psi.parent(b)));
      // for(auto it : range1(length(psi)))
      //   {
      //   auto psidag = dag(psi(it));
      //   auto link = commonIndex(psi(it), psi(psi.parent(it)));
      //   psidag.replaceInds({link}, {sim(link)});
      //   PrintData(psi(it));
      //   PrintData(psidag);
      //   PrintData(psi(it) * psidag);
      //   }

      // MPO Nop(SpinHalf(length(psi),{"ConserveQNs",false}));
      // for(auto n : range1(length(Nop)))
      //   {
      //   Nop.ref(n).replaceInds({Nop(n).inds()[0], Nop(n).inds()[1]}, {PH.H()(n).inds()[0], PH.H()(n).inds()[1]});
      //   }
      // auto xdag = prime(dag(psi));
      // xdag.replaceLinkInds(sim(linkInds(xdag)));
      // auto N_sites = length(psi);
      // auto height = intlog2(N_sites) - 1;
      // std::vector<ITensor> yAx(N_sites + 2);
      // for(auto n : range1(N_sites))
      //   {
      //   yAx[n] = Nop(n);
      //   }
      // for(int i = height; i >= 0; --i)
      //   {
      //   for(auto n : range1(pow2(i)))
      //     {
      //     if(n + pow2(i) - 2 == b || n + pow2(i) - 2 == psi.parent(b))
      //       {
      //       yAx[n] = yAx[2 * n - 1] * yAx[2 * n];
      //       }
      //     else
      //       {
      //       yAx[n] = psi(n + pow2(i) - 2) * yAx[2 * n - 1] * yAx[2 * n] * xdag(n + pow2(i) - 2);
      //       }
      //     }
      //   }
      // PrintData(yAx[1]);
      if(subspace_exp && psi.parent(b) >= 0)//Do subspace expansion only if there is link to be expansed
      {
        long current_dim=subspace_expansion(psi,PH,b,psi.parent(b),alpha);// We choose to put the zero into the parent
        args.add("MinDim",current_dim);
        orthPair(psi.ref(ha==1?b:psi.parent(b)),psi.ref(ha==1?psi.parent(b):b),args);
        psi.setOrthoLink(ha==1?b:psi.parent(b),ha==1?psi.parent(b):b); // Update orthogonalization
      }
      TIMER_STOP(4);

            if(!quiet)
        { 
                printfln("    Truncated to Cutoff=%.1E, Min_dim=%d, Max_dim=%d",
       sweeps.cutoff(sw),
       sweeps.mindim(sw), 
       sweeps.maxdim(sw) );
                printfln("    Trunc. err=%.1E, States kept: %s",
                         spec.truncerr(),
                         showDim(linkIndex(psi,b)) );
        }

            obs.lastSpectrum(spec);

            args.add("AtBond",b);
            args.add("HalfSweep",ha);
            args.add("Energy",energy); 
            args.add("Truncerr",spec.truncerr()); 

            obs.measure(args);

            // printfln("%d %d %d", sw, b, energy);

    } //for loop over b

        if(!silent)
    {
            auto sm = sw_time.sincemark();
            printfln("    Sweep %d/%d CPU time = %s (Wall time = %s)",
         sw,sweeps.nsweep(),showtime(sm.time),showtime(sm.wall));
#ifdef COLLECT_TIMES
            println(timers());
            timers().reset();
#endif
    }

        if(obs.checkDone(args)) break;
    
      } //for loop over sw
    psi.normalize();

    return energy;
  }

} //namespace itensor


#endif
