#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(int argc, char** argv)
{
  if(argc != 4) Error("Incorrect number of arguments!");

  int L = atoi(argv[1]);
  Real h = 1.0;
  auto sites = SpinHalf(L,{"ConserveQNs",false});

  int maxdim = atoi(argv[2]);
  Real co = atof(argv[3]);

  auto state = InitState(sites);
  for(auto i : range1(L))
    {
    if(i%2 == 1) state.set(i,"Up");
    else state.set(i,"Dn");
    }
  auto psi0 = BinaryTree(state);

  auto ampo = AutoMPO(sites);
  for(auto j : range1(L))
    {
    ampo += -4.0,"Sx",j,"Sx",j%L+1;
    ampo += -2.0*h,"Sz",j;
    }
  auto H = toMPO(ampo);

  // auto Hfull = H(1) * H(2) * H(3) * H(4);
  // auto inds = Hfull.inds();
  // auto C = std::get<0>(combiner(inds[0], inds[2], inds[4], inds[6]));
  // auto Cp = std::get<0>(combiner(inds[1], inds[3], inds[5], inds[7]));
  // auto Hfullmat = C * Hfull * Cp;
  // PrintData(Hfullmat);
  // Matrix M(std::pow(2, L), std::pow(2, L));
  // for(auto it : iterInds(Hfullmat)) {
  //   M(it[0].val - 1, it[1].val - 1) = Hfullmat.real(it);
  // }
  // Print(M);

  printfln("Initial norm = %.5f", std::real(innerC(psi0,psi0)));
  printfln("Initial energy = %.5f", std::real(innerC(psi0,H,psi0)));

  auto sweeps = Sweeps(30);
  sweeps.maxdim() = maxdim;
  sweeps.cutoff() = co;
  sweeps.niter() = 10;
  sweeps.noise() = 0.0;
  sweeps.alpha() = 0.1;
  println(sweeps);

  auto psi1 = std::get<1>(tree_dmrg(H,psi0,sweeps,{"NumCenter",1,"WhichEig","LargestReal","Quiet",}));

  printfln("\nFinal norm = %.5f", std::real(innerC(psi1,psi1)));
  printfln("\nGround state energy = %.10f", std::real(innerC(psi1,H,psi1)));

  for(auto j : range(psi1.size()))
    {
    println(j);
    PrintData(psi1(j).inds());
    }

  return 0;
}
