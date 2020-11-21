#ifndef __ITENSOR_TDVP_H
#define __ITENSOR_TDVP_H

#include "itensor/iterativesolvers.h"
#include "itensor/mps/localmposet.h"
#include "itensor/mps/sweeps.h"
#include "itensor/ttn/TreeDMRGObserver.h"
#include "itensor/util/cputime.h"


namespace itensor {

  template <class LocalOpT>
  Real
  TreeTDVPWorker(BinaryTree & psi,
	     LocalOpT& PH,
	     Cplx t,
	     const Sweeps& sweeps,
	     const Args& args = Args::global());

  template <class LocalOpT>
  Real
  TreeTDVPWorker(BinaryTree & psi,
	     LocalOpT& PH,
	     Cplx t,
	     const Sweeps& sweeps,
	     TreeDMRGObserver & obs,
	     Args args = Args::global());

  //
  // Available TDVP methods:
  // second order integrator: sweep left-to-right and right-to-left
  //

  //
  //TDVP with an MPO
  //
  Real inline
  tree_tdvp(BinaryTree & psi, 
       MPO const& H,
       Cplx t, 
       const Sweeps& sweeps,
       const Args& args = Args::global())
  {
    LocalMPO_BT PH(H,args);
    Real energy = TreeTDVPWorker(psi,PH,t,sweeps,args);
    return energy;
  }

  std::tuple<Real,BinaryTree> inline
  tree_tdvp(MPO const& H,
      BinaryTree const& psi0,
      Cplx t,
      Sweeps const& sweeps,
      Args const& args = Args::global())
  {
    auto psi = psi0;
    auto energy = tree_tdvp(psi,H,t,sweeps,args);
    return std::tuple<Real,BinaryTree>(energy,psi);
  }

  //
  //TDVP with an MPO and custom TreeDMRGObserver
  //
  Real inline
  tree_tdvp(BinaryTree & psi, 
       MPO const& H, 
       Cplx t,
       const Sweeps& sweeps, 
       TreeDMRGObserver & obs,
       const Args& args = Args::global())
  {
    LocalMPO_BT PH(H,args);
    Real energy = TreeTDVPWorker(psi,PH,t,sweeps,obs,args);
    return energy;
  }

  //
  // TreeTDVPWorker
  //

  template <class LocalOpT>
  Real
  TreeTDVPWorker(BinaryTree & psi,
	     LocalOpT& PH,
	     Cplx t,
	     Sweeps const& sweeps,
	     Args const& args)
  {
    TreeDMRGObserver obs(psi,args);
    Real energy = TreeTDVPWorker(psi,PH,t,sweeps,obs,args);
    return energy;
  }

  template <class LocalOpT>
  Real
  TreeTDVPWorker(BinaryTree & psi,
	     LocalOpT& H,
	     Cplx t,
	     Sweeps const& sweeps,
	     TreeDMRGObserver& obs,
	     Args args)
  { 
    // Truncate blocks of degenerate singular values (or not)
    args.add("RespectDegenerate",args.getBool("RespectDegenerate",true));

    const bool silent = args.getBool("Silent",false);
    if(silent)
      {
        args.add("Quiet",true);
        args.add("PrintEigs",false);
        args.add("NoMeasure",true);
        args.add("DebugLevel",-1);
      }
    const bool quiet = args.getBool("Quiet",false);
    const int debug_level = args.getInt("DebugLevel",(quiet ? -1 : 0));
    const int numCenter = args.getInt("NumCenter",2);
    args.add("Truncate",args.getBool("Truncate",numCenter == 2));

    Real energy = NAN;

    psi.setOrder(args); // Choose sweep order

    const bool subspace_exp=args.getBool("SubspaceExpansion",true);
    Real alpha = 0.0;
    args.add("DebugLevel",debug_level);
    args.add("DoNormalize",true);
    args.add("UseSVD",true);
    args.add("InclRoot",false);

    for(int sw = 1; sw <= sweeps.nsweep(); ++sw)
      {
        cpu_time sw_time;
        args.add("Sweep",sw);
        args.add("NSweep",sweeps.nsweep());
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("MinDim",sweeps.mindim(sw));
        args.add("MaxDim",sweeps.maxdim(sw));
        args.add("MaxIter",sweeps.niter(sw));
 
        if(subspace_exp)
        {
          alpha = sweeps.alpha(sw);
        }

        if(!H.doWrite()
           && args.defined("WriteDim")
           && sweeps.maxdim(sw) >= args.getInt("WriteDim"))
	  {
            if(!quiet)
	      {
                println("\nTurning on write to disk, write_dir = ",
                        args.getString("WriteDir","./"));
	      }
  
            // psi.doWrite(true);
            H.doWrite(true,args);
	  }

        // 0, 1 and 2-site wavefunctions
        ITensor phi0,phi1;
        Spectrum spec;
        for(int b = psi.startPoint(args), ha = 1; ha <= 2; sweepnext(b,ha,psi,args))
	  {
            if(!quiet)
	      printfln("Sweep=%d, HS=%d, Bond=%d/%d",sw,ha,b,psi.size()-1);

        psi.position(b,args); //Orthogonalize with respect to b

            H.numCenter(numCenter);
            H.position(b,psi);

            if(numCenter == 2)
	      phi1 = psi(b)*psi(psi.parent(b));
            else if(numCenter == 1)
	      phi1 = psi(b);

            applyExp(H,phi1,t/2,args);

            if(args.getBool("DoNormalize",true))
	      phi1 /= norm(phi1);
   
            if(numCenter == 2)
            {
	      spec = psi.svdBond(ha==1?b:psi.parent(b),phi1,ha==1?psi.parent(b):b,H,args);
        H.haveBeenUpdated(b);
        H.haveBeenUpdated(psi.parent(b)); // To known that we need to update the environement tensor
        if(subspace_exp && psi.parent(b) >= 0)//Do subspace expansion only if there is link to be expansed
        {
          long current_dim=subspace_expansion(psi,H,b,psi.parent(b),alpha);// We choose to put the zero into the parent
          args.add("MinDim",current_dim);
          orthPair(psi.ref(ha==1?b:psi.parent(b)),psi.ref(ha==1?psi.parent(b):b),args);
          psi.setOrthoLink(ha==1?b:psi.parent(b),ha==1?psi.parent(b):b); // Update orthogonalization
        }
            }
            else if(numCenter == 1)
            {
	      if(ha == 1) psi.ref(b) = phi1;
        else        psi.ref(psi.parent(b)) = phi1;
        H.haveBeenUpdated(b);
            }

	    // Calculate energy
            ITensor H_phi1;
            H.product(phi1,H_phi1);
            energy = real(eltC(dag(phi1)*H_phi1));
 
            if((ha == 1 && b != psi.endPoint(args)) || (ha == 2 && b != psi.startPoint(args)))
	      {
                auto b1 = (numCenter == 2 && ha == 1 ? psi.parent(b) : b);
 
                if(numCenter == 2)
		  {
                    phi0 = psi(b1);
		  }
                else if(numCenter == 1)
		  {
                    Index l;
                    l = commonIndex(psi(b),psi(psi.parent(b)));
                    ITensor U,S,V(l);
                    spec = svd(phi1,U,S,V,args);
                    if(ha == 1) psi.ref(b) = U;
                    else        psi.ref(psi.parent(b)) = U;
                    phi0 = S*V;
		  }
 
                H.numCenter(numCenter-1);
                H.position(b1,psi);
 
                applyExp(H,phi0,-t/2,args);
 
                if(args.getBool("DoNormalize",true))
		  phi0 /= norm(phi0);
                
                if(numCenter == 2)
		  {
                    psi.ref(b1) = phi0;
		  }
                if(numCenter == 1)
		  {
                    psi.ref(b1) *= phi0;
		  }

                if (numCenter == 1) H.haveBeenUpdated(b);
 
                // Calculate energy
                ITensor H_phi0;
                H.product(phi0,H_phi0);
                energy = real(eltC(dag(phi0)*H_phi0));
	      }
 
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
	  }
        
        if(obs.checkDone(args)) break;
  
      } //for loop over sw
  
    if(args.getBool("DoNormalize",true))
      {
        if(numCenter==1) psi.position(psi.startPoint());
        psi.normalize();
      }

    return energy;
  }

} //namespace itensor

#endif
