#include "itensor/all.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/ttn/tree_dmrg.h"
#include "itensor/ttn/tree_tdvp.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(int argc, char** argv)
{

// for(double lc = atof(argv[1]); lc <= atof(argv[2]); lc += atof(argv[3]))
// {
// for(double lo = atof(argv[1]); lo <= atof(argv[2]); lo += atof(argv[3]))
// {

  int N = 8;
  auto sites = SpinHalf(N,{"ConserveQNs",true});

  auto state = InitState(sites);
  for(auto i : range1(N)) // Note: sites are labelled from 1
    {
      if(i%2 == 1) state.set(i,"Up");
      else         state.set(i,"Dn");
    }
    for(auto i : range1(N/2)) // Note: sites are labelled from 1
      {
        state.set(i,"Up");
      }


  // auto ampo = AutoMPO(sites);
  // for(auto j : range1(N-1))
  //   {
  //     ampo += 0.5,"S+",j,"S-",j+1;
  //     ampo += 0.5,"S-",j,"S+",j+1;
  //     ampo +=     "Sz",j,"Sz",j+1;
  //   }
  // auto H = toMPO(ampo);
  // // H.swapSiteInds();

  auto anop = AutoMPO(sites);
  for(auto j : range1(N))
    {
      anop +=     "Sz",j;
    }
  auto Nop = toMPO(anop);

  // Real lc = atof(argv[1]), lo = atof(argv[2]);
  // printfln("lc %d lo %d", lc, lo);
  // // std::vector<Real> plist(N + 1);
  // // std::vector<Real> qlist(N + 1);
  // // for(auto j : range(N + 1))
  // // {
  // //   plist[j] = (Real)std::rand() / RAND_MAX;
  // //   qlist[j] = (Real)std::rand() / RAND_MAX;
  // // }
  // std::vector<Real> plist(N + 1, 0.1);
  // std::vector<Real> qlist(N + 1, 0.9);
  // plist[0] = plist[N] = qlist[0] = qlist[N] = 0.5;
  // auto ampo = AutoMPO(sites);
  // ampo += plist[0] * std::exp(lc), "S+", 1;
  // ampo += -plist[0], "projDn", 1;
  // ampo += qlist[0] * std::exp(-lc), "S-", 1;
  // ampo += -qlist[0], "projUp", 1;
  // for(auto j : range1(N - 1))
  // {
  //   ampo += plist[j] * std::exp(lc), "S-", j, "S+", j + 1;
  //   ampo += -plist[j], "projUp", j, "projDn", j + 1;
  //   ampo += qlist[j] * std::exp(-lc), "S+", j, "S-", j + 1;
  //   ampo += -qlist[j], "projDn", j, "projUp", j + 1;
  // }
  // ampo += qlist[N] * std::exp(-lc), "S+", N;
  // ampo += -qlist[N], "projDn", N;
  // ampo += plist[N] * std::exp(lc), "S-", N;
  // ampo += -plist[N], "projUp", N;
  // for(auto j : range1(N))
  // {
  //   ampo += -lo, "projDn", j;
  // }
  // auto H = toMPO(ampo);

  Real lc = atof(argv[1]), lo = atof(argv[2]);
  printfln("lc %d lo %d", lc, lo);
  // std::vector<Real> plist(N);
  // std::vector<Real> qlist(N);
  // for(auto j : range(N))
  // {
  //   plist[j] = (Real)std::rand() / RAND_MAX;
  //   qlist[j] = (Real)std::rand() / RAND_MAX;
  // }
  // std::vector<Real> plist(N, 0.6);
  // std::vector<Real> qlist(N, 0.4);
  std::vector<Real> plist({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 });
  std::vector<Real> qlist({ 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 });
  auto ampo = AutoMPO(sites);
  for(auto j : range1(N - 1))
  {
    ampo += plist[j - 1] * std::exp(lc), "S-", j, "S+", j + 1;
    ampo += -plist[j - 1], "projUp", j, "projDn", j + 1;
    ampo += qlist[j - 1] * std::exp(-lc), "S+", j, "S-", j + 1;
    ampo += -qlist[j - 1], "projDn", j, "projUp", j + 1;
  }
  ampo += plist[N - 1] * std::exp(lc), "S-", N, "S+", 1;
  ampo += -plist[N - 1], "projUp", N, "projDn", 1;
  ampo += qlist[N - 1] * std::exp(-lc), "S+", N, "S-", 1;
  ampo += -qlist[N - 1], "projDn", N, "projUp", 1;
  for(auto j : range1(N))
  {
    ampo += -lo, "projDn", j;
  }
  auto H = toMPO(ampo);

  // auto H = MPO_ASEP(sites, std::vector<Real>(N + 1, 0.6), std::vector<Real>(N + 1, 0.4), 0.0001);
  // auto H = MPO_ASEP(sites, std::vector<Real>(N, 0.6), std::vector<Real>(N, 0.4), 0.0);
  // auto H = MPO_ASEP(sites, { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }, { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 }, 0.0001);
  // auto H = MPO_ASEP(sites, { 0.1, 0.2, 0.3, 0.4 }, { 1.1, 1.2, 1.3, 1.4 }, 0.0);

  // auto Hfull = H(1) * H(2) * H(3) * H(4) * H(5) * H(6) * H(7) * H(8);
  // auto inds = Hfull.inds();
  // auto C = std::get<0>(combiner(inds[0], inds[2], inds[4], inds[6], inds[8], inds[10], inds[12], inds[14]));
  // auto Cp = std::get<0>(combiner(inds[1], inds[3], inds[5], inds[7], inds[9], inds[11], inds[13], inds[15]));
  // auto Hfullmat = C * Hfull * Cp;
  // PrintData(Hfullmat);
  // Matrix M(pow2(N), pow2(N));
  // for(auto it : iterInds(Hfullmat)) {
  //   M(it[0].val - 1, it[1].val - 1) = Hfullmat.real(it);
  // }
  // Print(M);

  println("Construction of the MPO and LocalMPO");
//  LocalMPO_BT PH(H,args);
//

  auto psi0 = BinaryTree(state);
  // PrintData(psi0);
  // auto psi0 = randomBinaryTree(sites, 100);

  // auto psidag = prime(dag(psi0));
  // std::vector<std::vector<ITensor>> MPO(psi0.height() + 2);
  //println(totalQN(psi0));
  println("Construction of the BinaryTree");
//  PH.position(0,psi0);
//  PH.haveBeenUpdated(2);
//  //println("----------New-------");
//  PH.position(3,psi0);
//removeQNs(psi0);
  printfln("Initial norm = %.5f", inner(psi0,psi0) );

  // inner calculates matrix elements of MPO's with respect to MPS's
  // inner(psi,H,psi) = <psi|H|psi>

  printfln("Initial energy = %.5f", inner(psi0,H,psi0) );
  printfln("Initial spin = %.5f", inner(psi0,Nop,psi0) );

  // Set the parameters controlling the accuracy of the DMRG
  // calculation for each DMRG sweep.
  // Here less than 5 cutoff values are provided, for example,
  // so all remaining sweeps will use the last one given (= 1E-10).
  //
  auto sweeps = Sweeps(8);
  sweeps.maxdim() = 5,10,15,16,16,16,16,16;
  // sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
  sweeps.cutoff() = 1E-13;
  sweeps.niter() = 10;
  sweeps.noise() = 0.0;
  sweeps.alpha() = 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1;
  //sweeps.noise() = 1E-7,1E-8,0.0; // The noise feature does not work for now
  println(sweeps);

  //
  // Begin the DMRG calculation
  //

  println("Start DMRG");
  // auto [energy1,psi1] = tree_dmrg(H,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"SubspaceExpansion",true,"DoSVDBond"});
  auto [energy1,psi1] = tree_dmrg(H,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"SubspaceExpansion",true,"WhichEig","LargestReal","DoSVDBond"});

  // auto [energy1,psi1] = tree_dmrg(H,psi0,sweeps,{"NumCenter",1,"Order","Default","Quiet",});

  //
  // Print the final energy reported by DMRG
  //
  printfln("\nFinal norm = %.5f", inner(psi1,psi1) );
  printfln("\nGround State Energy = %.10f",energy1);
  printfln("\nUsing inner = %.10f", inner(psi1,H,psi1) );
  printfln("Final spin = %.5f", inner(psi1,Nop,psi1) );
  // println(psi1);
  println(totalQN(psi1));

  // auto sweeps1 = Sweeps(2);
  // sweeps1.maxdim() = 16,16;
  // sweeps1.cutoff() = 1E-13;
  // sweeps1.niter() = 10;
  // sweeps1.noise() = 0.0;
  
  // println("\nStart TDVP");
  // using namespace std::complex_literals;
  // auto [energy2,psi2] = tree_tdvp(H,psi1,1.0e-4i,sweeps1,{"NumCenter",2,"Order","PostOrder","Quiet",});

  // printfln("\nFinal norm = %.5f", innerC(psi2,psi2) );
  // printfln("\nEnergy of Evolved State = %.10f",energy2);
  // printfln("\nUsing inner = %.10f", innerC(psi2,H,psi2) );
  // printfln("Final spin = %.5f", inner(psi2,Nop,psi2) );
  // println(psi2);
  // println(totalQN(psi2));
// }
// }

  return 0;
}
