#include "itensor/all.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/ttn/tree_dmrg.h"

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

  int N = 32;
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
  auto H = MPO_ASEP(sites, std::vector<Real>(N + 1, 0.6), std::vector<Real>(N + 1, 0.4), 0.0);

  println("Construction of the MPO and LocalMPO");
//  LocalMPO_BT PH(H,args);
//	

  auto psi0 = BinaryTree(state);
  // auto psi0 = randomBinaryTree(sites, 100);
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
  auto sweeps = Sweeps(20);
  sweeps.maxdim() = 10,10,10,10,10,20,20,20,20,20,100,100,100,100,100,200,200,200,200,200;
  sweeps.cutoff() = 1E-13;
  sweeps.niter() = 2;
  sweeps.noise() = 0.0;
  //sweeps.noise() = 1E-7,1E-8,0.0; // The noise feature does not work for now
  println(sweeps);

  //
  // Begin the DMRG calculation
  //

  println("Start DMRG");
  auto [energy,psi] = tree_dmrg(H,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",});
  // auto [energy,psi] = tree_dmrg(H,psi0,sweeps,{"NumCenter",1,"Order","Default","Quiet",});

  //
  // Print the final energy reported by DMRG
  //
  printfln("\nGround State Energy = %.10f",energy);
  printfln("\nUsing inner = %.10f", inner(psi,H,psi) );

  return 0;
}


