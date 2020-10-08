#include "itensor/all.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/ttn/tree_dmrg.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

void printVect(std::vector <int[2]> const &a) {
  std::cout << "The vector elements are : ";

  for(unsigned int i=0; i < a.size(); i++)
    std::cout <<'('<< a.at(i)[0]<<";" <<a.at(i)[1]<<')' << ' ';
  std::cout<< '\n';
}

void printVect(std::vector <int> const &a) {
  std::cout << "The vector elements are : ";

  for(unsigned int i=0; i < a.size(); i++)
    std::cout <<a.at(i)<<' ';

  std::cout<< '\n';
}

int main()
{

  int N = 8;
  auto sites = SpinHalf(N,{"ConserveQNs",false});

  auto state = InitState(sites);
  for(auto i : range1(N)) // Note: sites are labelled from 1
    {
      if(i%2 == 1) state.set(i,"Up");
      else         state.set(i,"Dn");
    }

  // auto ampo = AutoMPO(sites);
  // for(auto j : range1(N-1))
  //   {
  //     ampo += 0.5,"S+",j,"S-",j+1;
  //     ampo += 0.5,"S-",j,"S+",j+1;
  //     ampo +=     "Sz",j,"Sz",j+1;
  //   }
  // auto H = toMPO(ampo);

  // Real lambda = 0.0;
  // // std::vector<Real> plist(N + 1);
  // // std::vector<Real> qlist(N + 1);
  // // for(auto j : range(N + 1))
  // // {
  // //   plist[j] = (Real)std::rand() / RAND_MAX;
  // //   qlist[j] = (Real)std::rand() / RAND_MAX;
  // // }
  // std::vector<Real> plist(N + 1, 0.6);
  // std::vector<Real> qlist(N + 1, 0.4);
  // auto ampo = AutoMPO(sites);
  // ampo += plist[0] * std::exp(lambda), "S-", 1;
  // ampo += -plist[0], "projDn", 1;
  // ampo += qlist[0] * std::exp(-lambda), "S+", 1;
  // ampo += -qlist[0], "projUp", 1;
  // for(auto j : range1(N - 1))
  // {
  //   ampo += plist[j] * std::exp(lambda), "S+", j, "S-", j + 1;
  //   ampo += -plist[j], "projUp", j, "projDn", j + 1;
  //   ampo += qlist[j] * std::exp(-lambda), "S-", j, "S+", j + 1;
  //   ampo += -qlist[j], "projDn", j, "projUp", j + 1;
  // }
  // ampo += qlist[N] * std::exp(-lambda), "S-", N;
  // ampo += -qlist[N], "projDn", N;
  // ampo += plist[N] * std::exp(lambda), "S+", N;
  // ampo += -plist[N], "projUp", N;
  // auto H = toMPO(ampo);

  // Real lambda = 0.0;
  // std::vector<Real> plist(N);
  // std::vector<Real> qlist(N);
  // for(auto j : range(N))
  // {
  //   plist[j] = (Real)std::rand() / RAND_MAX;
  //   qlist[j] = (Real)std::rand() / RAND_MAX;
  // }
  // // std::vector<Real> plist(N, 0.6);
  // // std::vector<Real> qlist(N, 0.4);
  // // std::vector<Real> plist({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
  // // std::vector<Real> qlist({ 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 });
  // auto ampo = AutoMPO(sites);
  // for(auto j : range1(N - 1))
  // {
  //   ampo += plist[j - 1] * std::exp(lambda), "S+", j, "S-", j + 1;
  //   ampo += -plist[j - 1], "projUp", j, "projDn", j + 1;
  //   ampo += qlist[j - 1] * std::exp(-lambda), "S-", j, "S+", j + 1;
  //   ampo += -qlist[j - 1], "projDn", j, "projUp", j + 1;
  // }
  // ampo += plist[N - 1] * std::exp(lambda), "S+", N, "S-", 1;
  // ampo += -plist[N - 1], "projUp", N, "projDn", 1;
  // ampo += qlist[N - 1] * std::exp(-lambda), "S-", N, "S+", 1;
  // ampo += -qlist[N - 1], "projDn", N, "projUp", 1;
  // auto H = toMPO(ampo);

  auto H = MPO_ASEP(sites, std::vector<Real>(N + 1, 0.6), std::vector<Real>(N + 1, 0.4), 0.0);
  // auto H = MPO_ASEP(sites, std::vector<Real>(N, 0.6), std::vector<Real>(N, 0.4), 0.0);
  // auto H = MPO_ASEP(sites, { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }, { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 }, 0.0001);
  // auto H = MPO_ASEP(sites, { 0.1, 0.2, 0.3, 0.4 }, { 1.1, 1.2, 1.3, 1.4 }, 0.0);

  // auto Hfull = H(1) * H(2) * H(3) * H(4) * H(5) * H(6) * H(7) * H(8);
  // auto inds = Hfull.inds();
  // auto C = std::get<0>(combiner(inds[0], inds[2], inds[4], inds[6], inds[8], inds[10], inds[12], inds[14]));
  // auto Cp = std::get<0>(combiner(inds[1], inds[3], inds[5], inds[7], inds[9], inds[11], inds[13], inds[15]));
  // auto Hfullmat = C * Hfull * Cp;
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

  // Set the parameters controlling the accuracy of the DMRG
  // calculation for each DMRG sweep.
  // Here less than 5 cutoff values are provided, for example,
  // so all remaining sweeps will use the last one given (= 1E-10).
  //
  auto sweeps = Sweeps(5);
  sweeps.maxdim() = 10,20,100,100,200;
  // sweeps.maxdim() = 10,10,10,10,10,20,20,20,20,20,100,100,100,100,100,200,200,200,200,200;
  sweeps.cutoff() = 1E-13;
  sweeps.niter() = 10;
  sweeps.noise() = 0.0;
  //sweeps.noise() = 1E-7,1E-8,0.0; // The noise feature does not work for now
  println(sweeps);

  //
  // Begin the DMRG calculation
  //

  println("Start DMRG");
  auto [energy,psi] = tree_dmrg(H,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"SubspaceExpansion",true});
  // auto [energy,psi] = tree_dmrg(H,psi0,sweeps,{"NumCenter",1,"Order","Default","Quiet",});

  //
  // Print the final energy reported by DMRG
  //
  printfln("\nGround State Energy = %.10f",energy);
  printfln("\nUsing inner = %.10f", inner(psi,H,psi) );

  return 0;
}
