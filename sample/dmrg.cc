#include "itensor/all.h"
#include "itensor/util/print_macro.h"
using namespace itensor;

int 
main()
    {
    int N = 64;

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
  std::vector<Real> plist(N + 1, 0.1);
  std::vector<Real> qlist(N + 1, 0.9);
  plist[0] = plist[N] = qlist[0] = qlist[N] = 0.5;
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

  // auto H = MPO_ASEP(sites, std::vector<Real>(N + 1, 0.6), std::vector<Real>(N + 1, 0.4), -0.0001);

  // PrintData(H);
  // auto Hfull = H(1) * H(2) * H(3) * H(4);
  // auto inds = Hfull.inds();
  // auto C = std::get<0>(combiner(inds[0], inds[2], inds[4], inds[6]));
  // auto Cp = std::get<0>(combiner(inds[1], inds[3], inds[5], inds[7]));
  // auto Hfullmat = C * Hfull * Cp;
  // Matrix M(std::pow(2, N), std::pow(2, N));
  // for(auto it : iterInds(Hfullmat)) {
  //   M(it[0].val - 1, it[1].val - 1) = Hfullmat.real(it);
  // }
  // Print(M);

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
    auto sweeps = Sweeps(30);
    // sweeps.maxdim() = 10,20,100,100,200;
    sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
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
