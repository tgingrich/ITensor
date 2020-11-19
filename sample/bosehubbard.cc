#include "itensor/all.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/ttn/tree_dmrg.h"
#include "itensor/ttn/tree_tdvp.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main()
{

  int N = 16;
  auto sites = SpinHalf(N,{"ConserveQNs",false});

  auto state = InitState(sites);
  for(auto i : range1(N)) // Note: sites are labelled from 1
    {
      if(i%2 == 1) state.set(i,"Up");
      else         state.set(i,"Dn");
    }

  auto ampo = AutoMPO(sites);
  for(auto j : range1(N-1))
    {
      ampo += 0.5,"S+",j,"S-",j+1;
      ampo += 0.5,"S-",j,"S+",j+1;
      ampo +=     "Sz",j,"Sz",j+1;
    }
  auto H = toMPO(ampo);

  auto psi0 = BinaryTree(state);

  printfln("Initial norm = %.5f", inner(psi0,psi0) );
  printfln("Initial energy = %.5f", inner(psi0,H,psi0) );
  printfln("Initial spin = %.5f", inner(psi0,Nop,psi0) );

  auto sweeps = Sweeps(8);
  sweeps.maxdim() = 16,16,16,16,16,16,16,16;
  // sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
  sweeps.cutoff() = 1E-13;
  sweeps.niter() = 100;
  sweeps.noise() = 0.0;
  sweeps.alpha() = 0.1,0.1,0.05,0.05,0.02,0.02,0.01,0.01;
  // sweeps.alpha() = 0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002;
  println(sweeps);

  println("Start DMRG");
  auto [energy1,psi1] = tree_dmrg(H,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"});

  printfln("\nFinal norm = %.5f", inner(psi1,psi1) );
  printfln("\nGround State Energy = %.10f",energy1);
  printfln("\nUsing inner = %.10f", inner(psi1,H,psi1) );
  printfln("Final spin = %.5f", inner(psi1,Nop,psi1) );

  auto sweeps1 = Sweeps(2);
  sweeps1.maxdim() = 16,16;
  // sweeps1.maxdim() = 300,300;
  sweeps1.cutoff() = 1E-13;
  sweeps1.niter() = 100;
  sweeps1.noise() = 0.0;
  sweeps.alpha() = 0.001,0.001;
  println(sweeps1);
  
  println("\nStart TDVP");
  using namespace std::complex_literals;
  auto [energy2,psi2] = tree_tdvp(H,psi1,0.1,sweeps1,{"NumCenter",2,"Order","PostOrder","Quiet",});

  printfln("\nFinal norm = %.5f", std::real(innerC(psi2,psi2)) );
  printfln("\nEnergy of Evolved State = %.10f",energy2);
  printfln("\nUsing inner = %.10f", std::real(innerC(psi2,H,psi2)) );
  printfln("Final spin = %.5f", std::real(innerC(psi2,Nop,psi2)) );

  return 0;
}
