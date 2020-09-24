#include "test.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/mps/sites/spinhalf.h"
#include "itensor/mps/sites/fermion.h"
#include "itensor/util/print_macro.h"
#include "itensor/util/str.h"
#include "itensor/ttn/tree_dmrg.h"
#include "itensor/mps/autompo.h"

namespace itensor {

bool inline
checkTags(BinaryTree const& psi,
         std::string const& siteTags = "Site",
         std::string const& linkTags = "Link")
{
 auto N = psi.size();
 auto L= psi.length();
 for(auto n : range1(N))
 {
     if( n < N )
       {
       auto l = linkIndex(psi,n);
       if( !hasTags(l,linkTags+",l="+str(n)) ) return false;
       }
 }
 for(auto n : range1(L))
 {
     auto s = siteIndex(psi,n);
     if( !hasTags(s,siteTags+",n="+str(n)) ) return false;
 }
 return true;
}

Real inline
diff(BinaryTree const& psi, BinaryTree const& phi)
  {
  auto norm_psi = real(innerC(psi,psi));
  auto norm_phi = real(innerC(phi,phi));
  auto psi_phi = innerC(psi,phi);
  return std::sqrt(std::abs(norm_psi/norm_phi+1.-2.*real(psi_phi)/norm_phi));
  }
}


using namespace itensor;
using std::vector;

TEST_CASE("BTTest")
{

static const int N = 8;

// SiteSet with no QNs
auto shsites = SpinHalf(N,{"ConserveQNs=",false});
auto shFerro = InitState(shsites,"Up");
auto shNeel = InitState(shsites);
for(auto j : range1(N))
    shNeel.set(j,j%2==1 ? "Up" : "Dn");


auto shsitesQNs = SpinHalf(N,{"ConserveQNs=",true});
auto shFerroQNs = InitState(shsitesQNs,"Up");
auto shNeelQNs = InitState(shsitesQNs);
for(auto j : range1(N))
    shNeelQNs.set(j,j%2==1 ? "Up" : "Dn");

SECTION("QNCheck")
    {
    auto psiNeelQNs = BinaryTree(shNeelQNs);

    CHECK(hasQNs(psiNeelQNs));

    CHECK(checkQNs(psiNeelQNs));
    CHECK_EQUAL(totalQN(psiNeelQNs),QN({"Sz",0}));

    auto psiFerroQNs = BinaryTree(shFerroQNs);

    CHECK(checkQNs(psiFerroQNs));
    CHECK_EQUAL(totalQN(psiFerroQNs),QN({"Sz",8}));
    }

SECTION("hasQNs")
    {
    CHECK(hasQNs(shFerroQNs));
    CHECK(not hasQNs(shFerro));
    }

SECTION("Constructors (m==1)")
    {
    auto psi = BinaryTree(shsitesQNs);
    CHECK(1==dim(linkIndex(psi,3)));//TODO
    CHECK(checkTags(psi));
    }

SECTION("Constructors (dim>1)")
    {
    auto d = 4;
    auto psi = BinaryTree(shsites,d);
    auto l2 = commonIndex(psi(1),psi(3));
    CHECK(d==dim(l2));

    CHECK(checkTags(psi));

    psi.randomize({"Complex=",true});

    psi.position(1);

    CHECK(checkTags(psi));
    CHECK(isOrtho(psi));

    psi.position(N-2);

    CHECK(checkTags(psi));
    CHECK(isOrtho(psi));

    psi.normalize();

    CHECK_CLOSE(norm(psi),1.);

    }

SECTION("Random constructors (dim==1)")
    {
    auto psi = randomBinaryTree(shsites);
    psi.ref(1) *= Complex_i;

    CHECK(checkTags(psi));

    auto psi1 = psi;
    psi1.position(1);

    auto norm2 = innerC(psi1,psi1);

    CHECK_CLOSE(norm(psi1),std::sqrt(real(norm2)));
    CHECK_CLOSE(imag(norm2),0.);
    CHECK(checkTags(psi1));
    CHECK_CLOSE(diff(psi,psi1),0.);

    psi1.position(N-2);

    norm2 = innerC(psi1,psi1);

    CHECK_CLOSE(norm(psi1),std::sqrt(real(norm2)));
    CHECK_CLOSE(imag(norm2),0.);
    CHECK(checkTags(psi1));
    CHECK_CLOSE(diff(psi,psi1),0.);
    }

SECTION("Check .position() with custom tags")
    {
    auto psi = randomBinaryTree(shNeelQNs);
    psi.ref(1) *= Complex_i;

    CHECK(checkTags(psi,"Site","Link"));

    psi.replaceTags("Site","MySite");
    psi.replaceTags("Link","MyLink");

    CHECK(checkTags(psi,"MySite","MyLink"));

    auto opsi = psi;

    opsi.position(1);

    auto norm_opsi = std::sqrt(real(innerC(opsi,opsi)));

    // CHECK(checkOrtho(opsi));
    CHECK_CLOSE(norm(opsi),norm_opsi);
    CHECK_CLOSE(diff(opsi,psi),0.);
    CHECK(checkTags(opsi,"MySite","MyLink"));

    opsi.position(N-2);

    CHECK_CLOSE(diff(opsi,psi),0.);
    CHECK(checkTags(opsi,"MySite","MyLink"));
    }

SECTION("Check .orthogonalize() with custom tags")
    {
    auto psi = randomBinaryTree(shNeelQNs);
    psi.ref(1) *= Complex_i;

    CHECK(checkTags(psi,"Site","Link"));

    psi.replaceTags("Site","MySite");
    psi.replaceTags("Link","MyLink");

    CHECK(checkTags(psi,"MySite","MyLink"));

    auto opsi = psi;

    opsi.orthogonalize();

    // CHECK(checkOrtho(opsi));
    CHECK_CLOSE(norm(opsi),std::sqrt(real(innerC(opsi,opsi))));
    CHECK_CLOSE(diff(opsi,psi),0.);
    CHECK(checkTags(opsi,"MySite","MyLink"));
    }

SECTION("Random constructors, QN conserved (dim==1)")
    {
    auto psi = randomBinaryTree(shFerroQNs);
    psi.ref(1) *= Complex_i;
    psi.orthogonalize();
    CHECK(norm(psi)>0);
    CHECK(checkTags(psi));
    }

// SECTION("MPSAddition 1")
//     {
//     auto sites = Fermion(10);
//
//     auto i1 = InitState(sites,"Emp");
//     auto i2 = i1;
//
//     i1.set(1,"Occ");
//     i2.set(2,"Occ");
//
//     //"Valence bond" between sites 1 and 2
//     auto psi = ISqrt2*sum(MPS(i1),MPS(i2));
//
//     CHECK(checkTags(psi));
//     CHECK_CLOSE(norm(psi),1);
//     CHECK_EQUAL(totalQN(psi),QN({"Nf",1}));
//     }
//
// SECTION("MPSAddition Custom Tags")
//     {
//     auto sites = Fermion(10,{"ConserveQNs=",true});
//
//     auto i1 = InitState(sites,"Emp");
//     auto i2 = i1;
//
//     i1.set(1,"Occ");
//     i2.set(2,"Occ");
//
//     auto psi1 = MPS(i1);
//     psi1.ref(1) *= Complex_i;
//     auto psi2 = MPS(i2);
//     psi2.ref(1) *= Complex_i;
//
//     psi1.replaceTags("Site","MySite").replaceTags("Link","MyLink1");
//     psi2.replaceTags("Site","MySite").replaceTags("Link","MyLink2");
//
//     //"Valence bond" between sites 1 and 2
//     auto psi12 = ISqrt2*sum(psi1,psi2);
//
//     CHECK(checkTags(psi12,"MySite","MyLink1"));
//     CHECK_CLOSE(norm(psi12),1);
//     CHECK_EQUAL(totalQN(psi12),QN({"Nf",1}));
//
//     auto psi21 = ISqrt2*sum(psi2,psi1);
//     CHECK(checkTags(psi21,"MySite","MyLink2"));
//     CHECK_CLOSE(norm(psi21),1);
//     CHECK_EQUAL(totalQN(psi21),QN({"Nf",1}));
//     }
//
// SECTION("MPSAddition 2")
//     {
//     auto sites = SiteSet(10,2);
//     auto psi1 = randomMPS(sites);
//     psi1.ref(1) *= Complex_i;
//     auto psi2 = randomMPS(sites);
//     psi2.ref(1) *= Complex_i;
//     auto psi = sum(psi1,psi2);
//
//     auto exact_inner = innerC(psi1,psi2);
//     exact_inner += innerC(psi2,psi1);
//     exact_inner += innerC(psi1,psi1);
//     exact_inner += innerC(psi2,psi2);
//
//     CHECK(checkTags(psi));
//     CHECK_CLOSE(exact_inner,innerC(psi,psi));
//     CHECK_EQUAL(order(psi(1)),2);
//     CHECK_EQUAL(order(psi(2)),3);
//     CHECK_EQUAL(order(psi(5)),3);
//     CHECK_EQUAL(order(psi(9)),3);
//     CHECK_EQUAL(order(psi(10)),2);
//     }

SECTION("PositionTest")
    {
    auto sites = Fermion(8);

    auto init = InitState(sites,"Emp");
    init.set(2,"Occ");
    init.set(4,"Occ");
    init.set(6,"Occ");

    auto psi = BinaryTree(init);
    psi.ref(1) *= Complex_i;

    psi.position(1,{"Cutoff=",1E-8});

    // CHECK(checkOrtho(psi));
    CHECK_EQUAL(findCenter(psi),1);

    psi.position(4,{"Cutoff=",1E-8});

    CHECK_EQUAL(findCenter(psi),4);
    }

SECTION("Orthogonalize")
    {
    auto d = 20;
    auto sites = SpinHalf(N,{"ConserveQNs=",false});

    //Make a random BinaryTree of bond dim. d
    auto psi = BinaryTree(sites,d);
    psi.randomize({"Complex=",true});

    CHECK(checkTags(psi));

    //Normalize psi
    auto n2 = real(innerC(psi,psi));
    psi.ref(1) /= sqrt(n2);

    auto opsi = psi;
    opsi.orthogonalize({"Cutoff",1E-16});

    // CHECK(checkOrtho(opsi));
    CHECK_CLOSE(innerC(opsi,psi),1.0);
    CHECK(checkTags(opsi));

    psi.orthogonalize({"MaxDim=",10,"Cutoff=",1E-16});

    // CHECK(checkOrtho(psi));
    CHECK(maxLinkDim(psi)==10);
    CHECK(checkTags(psi));
    }

// SECTION("Overlap - 1 site")
//     {
//     auto psi = BinaryTree(1);
//     auto s = Index(2,"s");
//     psi.ref(1) = randomITensorC(s);
//
//     CHECK_CLOSE(innerC(psi,psi),eltC(dag(psi(1))*psi(1)));
//     }

SECTION("siteInds")
    {
    auto sites = SpinHalf(N);
    auto initstate = InitState(sites,"Up");
    auto psi = randomBinaryTree(initstate);

    CHECK(checkTags(psi));

    auto s = siteInds(psi);

    for( auto n : range1(N) )
      {
      CHECK( siteIndex(psi,n)==s(n) );
      CHECK( sites(n)==s(n) );
      }
    }

SECTION("replaceSiteInds")
    {
    auto sites1 = SpinHalf(N);
    auto initstate1 = InitState(sites1,"Up");
    auto psi1 = randomBinaryTree(initstate1);
    psi1.position(1);
    psi1 /= norm(psi1);

    CHECK(checkTags(psi1));

    auto sites2 = SpinHalf(N);
    auto initstate2 = InitState(sites2,"Up");
    auto psi2 = randomBinaryTree(initstate2);
    psi2.position(1);
    psi2 /= norm(psi2);
    psi2.replaceTags("Site","MySite");

    CHECK(checkTags(psi2,"MySite","Link"));

    auto s1 = siteInds(psi1);
    auto s2 = siteInds(psi2);

    auto psi1_new = replaceSiteInds(psi1,s2);

    CHECK(checkTags(psi1_new,"MySite","Link"));
    for( auto n : range1(N) )
      CHECK( siteIndex(psi1_new,n)==siteIndex(psi2,n) );
    }

SECTION("prime")
    {
    auto s = SpinHalf(N,{"ConserveQNs=",false});
    auto psi = randomBinaryTree(s);

    auto psi1 = prime(psi);

    CHECK( prime(siteIndex(psi,3)) == siteIndex(psi1,3) );
    CHECK( prime(linkIndex(psi,3)) == linkIndex(psi1,3) );

    auto psi2 = prime(psi,"Site");

    CHECK( prime(siteIndex(psi,3)) == siteIndex(psi2,3) );
    CHECK( linkIndex(psi,3) == linkIndex(psi2,3) );

    auto psi3 = mapPrime(psi2,1,2);

    CHECK( prime(siteIndex(psi,3),2) == siteIndex(psi3,3) );
    CHECK( linkIndex(psi,3) == linkIndex(psi3,3) );

    auto psi4 = swapPrime(psi2,0,1);

    CHECK( siteIndex(psi,3) == siteIndex(psi4,3) );
    CHECK( prime(linkIndex(psi,3)) == linkIndex(psi4,3) );

    }

    SECTION("Tree DMRG")
      {
      int N = 32;
      auto sites = SpinHalf(N,{"ConserveQNs=",false});
      auto psi0 = randomBinaryTree(InitState(sites,"Up"));

      auto h = 0.5; // Critical point

      auto ampo = AutoMPO(sites);
      for(int j = 1; j < N; ++j)
          {
          ampo += -1.0,"Sx",j,"Sx",j+1;
          ampo += -h,"Sz",j;
          }
      ampo += -h,"Sz",N;
      auto H = toMPO(ampo);

      auto sweeps = Sweeps(5);
      sweeps.maxdim() = 10,20,30;
      sweeps.cutoff() = 1E-12;
      auto [Energy,psi] = tree_dmrg(H,psi0,sweeps,{"Order","PostOrder","Silent",true});
      auto energy = Energy/N;
      (void)psi;

      // Exact energy for transverse field Ising model
      // with open boundary conditions
      auto Energy_exact = 1.0 - 1.0/sin(Pi/(2*(2*N+1)));
      auto energy_exact = Energy_exact/(4*N);
      CHECK_CLOSE((energy-energy_exact)/energy_exact,0.);
      }

    SECTION("Tree DMRG with QNs")
      {
      int N = 32;
      auto sites = SpinHalf(N,{"ConserveSz=",false,
                               "ConserveParity=",true});
      auto psi0 = randomBinaryTree(InitState(sites,"Up"));

      auto h = 0.5; // Critical point

      auto ampo = AutoMPO(sites);
      for(int j = 1; j < N; ++j)
          {
          ampo += -1.0,"Sx",j,"Sx",j+1;
          ampo += -h,"Sz",j;
          }
      ampo += -h,"Sz",N;
      auto H = toMPO(ampo);

      auto sweeps = Sweeps(5);
      sweeps.maxdim() = 10,20,30;
      sweeps.cutoff() = 1E-12;
      auto [Energy,psi] = tree_dmrg(H,psi0,sweeps,{"Order","PostOrder","Silent",true});
      auto energy = Energy/N;
      (void)psi;

      // Exact energy for transverse field Ising model
      // with open boundary conditions
      auto Energy_exact = 1.0 - 1.0/sin(Pi/(2*(2*N+1)));
      auto energy_exact = Energy_exact/(4*N);
      CHECK_CLOSE((energy-energy_exact)/energy_exact,0.);
      }

}
