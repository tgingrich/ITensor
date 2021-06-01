#include <set>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
using namespace itensor;

void
gen_perms(std::set<std::vector<int>>& perms, std::vector<int>& a, int l, int r)
    {
    if(l==r) perms.insert(a);
    else
        {
        for(int i=l; i<=r; ++i)
            {
            std::swap(a[l],a[i]);
            gen_perms(perms,a,l+1,r);
            std::swap(a[l],a[i]);
            }
        }
    }

int 
main()
    {
    int N = 100;

    //
    // Initialize the site degrees of freedom
    // Setting "ConserveQNs=",true makes the indices
    // carry Sz quantum numbers and will lead to 
    // block-sparse MPO and MPS tensors
    //
    auto sites = SpinHalf(N,{"ConserveQNs",true}); //make a chain of N spin 1/2's
    // auto sites = SpinOne(N); //make a chain of N spin 1's

    //
    // Use the AutoMPO feature to create the 
    // next-neighbor Heisenberg model
    //
    auto ampo = AutoMPO(sites);
    for(auto j : range1(N-1))
        {
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
        ampo +=     "Sz",j,"Sz",j+1;
        }
    auto H = toMPO(ampo);

    // std::vector<AutoMPO> densops(N,sites);
    // std::set<std::vector<int>> perms;
    // for(auto n : range((N+1)/2))
    //     {
    //     std::vector<int> a;
    //     for(auto i : range(N-2*n-1)) a.push_back(0);
    //     for(auto i : range(n))
    //         {
    //         a.push_back(1);
    //         a.push_back(2);
    //         }
    //     gen_perms(perms,a,0,N-2);
    //     }
    // println(perms.size());
    // for(auto it = perms.begin(); it != perms.end(); ++it)
    //     {
    //     for(auto i : range1(N))
    //         {
    //         HTerm term;
    //         term.add("projDn",i);
    //         for(auto j : range1(N-1))
    //             {
    //             std::string op;
    //             switch(it->at(j-1))
    //                 {
    //                 case 1: op="S+"; break;
    //                 case 2: op="S-";
    //                 }
    //             if(op.size())
    //                 {
    //                 term.add(op,j<i?j:j+1);
    //                 }
    //             }
    //         densops[i-1].add(term);
    //         }
    //     }
    // for(auto densop : densops) 
    // PrintData(toMPO(densops[0]));
    // PrintData(toMPO(densops[N/4]));
    // PrintData(toMPO(densops[N/2]));
    // PrintData(toMPO(densops[3*N/4]));

  // Real lc = 3.0;
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
  // auto H = toMPO(ampo);

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
    // PrintData(psi0);

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
    auto sweeps = Sweeps(6);
    sweeps.maxdim() = 16,16,16,16,16,16;
    // sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
    sweeps.cutoff() = 1E-13;
    sweeps.niter() = 10;
    // sweeps.noise() = 1E-7,1E-8,0.0,0.0,0.0,0.0;
    sweeps.noise() = 0.0;
    println(sweeps);

    // //
    // // Begin the DMRG calculation
    // //
    auto [energy,psi] = dmrg(H,psi0,sweeps,{"NumCenter",2,"Quiet",true,"WhichEig","LargestReal"});

    auto psi1 = sum(psi,psi0);
    // PrintData(psi);
    // PrintData(psi0);
    // PrintData(psi1);

    //
    // Print the final energy reported by DMRG
    //
    printfln("\nGround State Energy = %.10f",energy);
    printfln("\nUsing inner = %.10f", inner(psi,H,psi) );

    auto sweeps1 = Sweeps(10);
    sweeps1.maxdim() = 16;
    sweeps1.cutoff() = 1E-13;
    sweeps1.niter() = 100;
    sweeps1.noise() = 0.0;
    println(sweeps1);

    // println("\nStart TDVP");
    // auto energy2 = tdvp(psi,H,0.1,sweeps1,{"NumCenter",1,"Quiet",});

    // printfln("\nEnergy of Evolved State = %.10f",energy2);

    psi = psi0;
    for(auto i : range1(10))
        {
        auto psi1 = psi;
        tdvp(psi,H,0.1,sweeps1,{"NumCenter",1,"DoNormalize",false,"Quiet",});
        printfln("\n%d: SCGF = %f",i,std::log(inner(psi1,psi)));
        psi.position(1);
        psi.normalize();
        }

    return 0;
    }
