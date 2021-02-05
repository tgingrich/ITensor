#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main()
{
  int L = 4;
  Real J = 1.0;
  Real Ui = 2*J;
  Real Uf = 7*J;
  Real phi = 0.7*M_PI;
  Real h = 1.0;
  Real gamma = J/(6*h);
  Real t0 = (1-Uf/Ui)/gamma;
  Real dt = 2.0E-3*h/J;
  Real c = 1.0;
  auto sites = Boson(L,{"MaxOcc",4,"ConserveQNs",true});

  auto state = InitState(sites);
  for(auto i : range1(L))
    {
    state.set(i,"1");
    }
  auto psi0 = BinaryTree(state);

  auto ampo = AutoMPO(sites);
  for(auto j : range1(L-1))
    {
    ampo += -c*J*std::exp(Cplx_i*phi/L),"A",j,"Adag",j+1;
    ampo += -c*J*std::exp(-Cplx_i*phi/L),"Adag",j,"A",j+1;
    }
  ampo += -c*J*std::exp(Cplx_i*phi/L),"A",L,"Adag",1;
  ampo += -c*J*std::exp(-Cplx_i*phi/L),"Adag",L,"A",1;
  for(auto j : range1(L))
    {
    ampo += Ui/2,"N",j,"N",j;
    ampo += -Ui/2,"N",j;
    }
  auto H = toMPO(ampo);

  auto acur = AutoMPO(sites);
  for(auto j : range1(L-1))
    {
    acur += Cplx_i*c*J/(h*L)*std::exp(Cplx_i*phi/L),"A",j,"Adag",j+1;
    acur += -Cplx_i*c*J/(h*L)*std::exp(-Cplx_i*phi/L),"Adag",j,"A",j+1;
    }
  acur += Cplx_i*c*J/(h*L)*std::exp(Cplx_i*phi/L),"A",L,"Adag",1;
  acur += -Cplx_i*c*J/(h*L)*std::exp(-Cplx_i*phi/L),"Adag",L,"A",1;
  auto I = toMPO(acur);

  // PrintData(psi0);
  // PrintData(H);
  // PrintData(I);

  printfln("Initial norm = %.5f", std::real(innerC(psi0,psi0)));
  printfln("Initial current = %.5f", std::real(innerC(psi0,I,psi0)));

  auto sweeps = Sweeps(10);
  sweeps.maxdim() = 10,20,30,40,50,60,60,60,60,60;
  sweeps.cutoff() = 1E-13;
  sweeps.niter() = 10;
  sweeps.noise() = 0.0;
  sweeps.alpha() = 0.1,0.1,0.05,0.05,0.02,0.02,0.01,0.01,0.005,0.005;
  println(sweeps);

  auto psi1 = std::get<1>(tree_dmrg(H,psi0,sweeps,{"NumCenter",2,"Quiet",}));

  printfln("\nFinal norm = %.5f", std::real(innerC(psi1,psi1)));
  printfln("\nGround state current = %.10f", std::real(innerC(psi1,I,psi1)));

  // auto Hfull = H(1) * H(2) * H(3) * H(4);
  // auto inds = Hfull.inds();
  // auto C = std::get<0>(combiner(inds[0], inds[2], inds[4], inds[6]));
  // auto Cp = std::get<0>(combiner(inds[1], inds[3], inds[5], inds[7]));
  // auto Hfullmat = C * Hfull * Cp;
  // PrintData(Hfullmat);

  // Hfull = I(1) * I(2) * I(3) * I(4);
  // inds = Hfull.inds();
  // C = std::get<0>(combiner(inds[0], inds[2], inds[4], inds[6]));
  // Cp = std::get<0>(combiner(inds[1], inds[3], inds[5], inds[7]));
  // auto Ifullmat = C * Hfull * Cp;
  // PrintData(Ifullmat);

  // Hfull = psi1(0) * psi1(1) * psi1(2);
  // inds = Hfull.inds();
  // C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3]));
  // auto psi1fullmat = C * Hfull;
  // PrintData(psi1fullmat);

  auto sweeps1 = Sweeps(1);
  sweeps1.maxdim() = 60;
  sweeps1.cutoff() = 1E-13;
  sweeps1.niter() = 100;
  sweeps1.noise() = 0.0;
  println(sweeps1);
  
  for(Real t = t0, U = Ui; t < 25.0; t += dt)
    {
    if(t < 0.0)
      {
      U += (Ui - Uf) * dt / t0;
      ampo = AutoMPO(sites);
      for(auto j : range1(L-1))
        {
        ampo += -c*J*std::exp(Cplx_i*phi/L),"A",j,"Adag",j+1;
        ampo += -c*J*std::exp(-Cplx_i*phi/L),"Adag",j,"A",j+1;
        }
      ampo += -c*J*std::exp(Cplx_i*phi/L),"A",L,"Adag",1;
      ampo += -c*J*std::exp(-Cplx_i*phi/L),"Adag",L,"A",1;
      for(auto j : range1(L))
        {
        ampo += U/2,"N",j,"N",j;
        ampo += -U/2,"N",j;
        }
      H = toMPO(ampo);
      }

    psi1 = std::get<1>(tree_tdvp(H,psi1,Cplx_i*dt,sweeps1,{"NumCenter",2,"Quiet",}));

    printfln("Current measurement %d %d %d", t, U, std::real(innerC(psi1,I,psi1)));
    }

  return 0;
}
