#include "itensor/all.h"
#include "itensor/util/print_macro.h"
using namespace itensor;

int 
main()
    {
    int N = 128;

    //
    // Initialize the site degrees of freedom
    // Setting "ConserveQNs=",true makes the indices
    // carry Sz quantum numbers and will lead to 
    // block-sparse MPO and MPS tensors
    //
    auto sites = SpinHalf(N,{"ConserveQNs",false}); //make a chain of N spin 1/2's
    // auto sites = SpinOne(N); //make a chain of N spin 1's

    //
    // Use the AutoMPO feature to create the 
    // next-neighbor Heisenberg model
    //
    // auto ampo = AutoMPO(sites);
    // for(auto j : range1(N-1))
    //     {
    //     ampo += 0.5,"S+",j,"S-",j+1;
    //     ampo += 0.5,"S-",j,"S+",j+1;
    //     ampo +=     "Sz",j,"Sz",j+1;
    //     }
    // auto H = toMPO(ampo);

    Real lambda = 0.0;
  // std::vector<Real> plist(N + 1);
  // std::vector<Real> qlist(N + 1);
  // for(auto j : range(N + 1))
  // {
  //   plist[j] = (Real)std::rand() / RAND_MAX;
  //   qlist[j] = (Real)std::rand() / RAND_MAX;
  // }
  std::vector<Real> plist(N + 1, 0.6);
  std::vector<Real> qlist(N + 1, 0.4);
  auto ampo = AutoMPO(sites);
  ampo += plist[0] * std::exp(lambda), "S-", 1;
  ampo += -plist[0], "projDn", 1;
  ampo += qlist[0] * std::exp(-lambda), "S+", 1;
  ampo += -qlist[0], "projUp", 1;
  for(auto j : range1(N - 1))
  {
    ampo += plist[j] * std::exp(lambda), "S+", j, "S-", j + 1;
    ampo += -plist[j], "projUp", j, "projDn", j + 1;
    ampo += qlist[j] * std::exp(-lambda), "S-", j, "S+", j + 1;
    ampo += -qlist[j], "projDn", j, "projUp", j + 1;
  }
  ampo += qlist[N] * std::exp(-lambda), "S-", N;
  ampo += -qlist[N], "projDn", N;
  ampo += plist[N] * std::exp(lambda), "S+", N;
  ampo += -plist[N], "projUp", N;
  auto H = toMPO(ampo);

    // Set the initial wavefunction matrix product state
    // to be a Neel state.
    //
    auto state = InitState(sites);
    for(auto i : range1(N))
        {
        if(i%2 == 1) state.set(i,"Up");
        else         state.set(i,"Dn");
        }
    auto psi0 = MPS(state);

    //
    // inner calculates matrix elements of MPO's with respect to MPS's
    // inner(psi,H,psi) = <psi|H|psi>
    //
    printfln("Initial energy = %.5f", inner(psi0,H,psi0) );

    //
    // Set the parameters controlling the accuracy of the DMRG
    // calculation for each DMRG sweep. 
    // Here less than 5 cutoff values are provided, for example,
    // so all remaining sweeps will use the last one given (= 1E-10).
    //
    auto sweeps = Sweeps(20);
    // sweeps.maxdim() = 10,20,100,100,200;
    sweeps.maxdim() = 10,10,10,10,10,20,20,20,20,20,100,100,100,100,100,200,200,200,200,200;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 10;
    sweeps.noise() = 1E-7,1E-8,0.0;
    println(sweeps);

    //
    // Begin the DMRG calculation
    //
    auto [energy,psi] = dmrg(H,psi0,sweeps,"Quiet");

    //
    // Print the final energy reported by DMRG
    //
    printfln("\nGround State Energy = %.10f",energy);
    printfln("\nUsing inner = %.10f", inner(psi,H,psi) );

    return 0;
    }
