#include <chrono>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(int argc, char** argv)
	{
	if (argc != 4)
		{
	    Error("Incorrect number of arguments!");
		}
	auto start = std::chrono::high_resolution_clock::now();
	auto freq = atof(argv[3]);
	int nparticles = atoi(argv[2]);
	std::ifstream ifs;
	ifs.open(argv[1]);
	std::string type, val;
	Real len = 0.0, strength = 0.0, ratio = 0.0, di = 0.0, nstages = 0.0;
	int bins = 0;
	int idx = 0;
	while(ifs >> type >> val)
		{
		switch(idx++)
			{
			case 0:
				len = std::stod(val);
			case 1:
				bins = std::stoi(val);
			case 2:
				strength = std::stod(val);
			case 3:
				ratio = std::stod(val);
			case 4:
				di = std::stod(val);
			case 5:
				nstages = std::stod(val);
			}
		}
	auto h = len / bins;
	ifs.close();
	ifs.open("coefs.in");
	std::vector<Real> coefs;
	while(ifs >> val)
		{
		coefs.push_back(std::stod(val));
		}
	ifs.close();

	auto sites = SpinHalf(bins,{"ConserveQNs",true});
	auto state = InitState(sites);
	int count = 0;
	for(auto i : range(bins)) // Note: sites are labelled from 1
		{
		if(i%(bins/nparticles)==0 && count++ < nparticles) state.set(i+1,"Up");
		else state.set(i+1,"Dn");
		}
	auto psi0 = BinaryTree(state);

	Real dz = 0.0001;
	std::vector<Real> plist1(bins), qlist1(bins), plist2(bins), qlist2(bins);
	for(auto j : range(bins))
		{
		plist1[j] = qlist1[j] = plist2[j] = qlist2[j] = di/std::pow(h,2);
		for (auto i : range((int)coefs.size()))
			{
    		plist1[j] += -(i+1)*strength*coefs[i]*M_PI*std::cos(2*(i+1)*M_PI*j*h)/h;
    		qlist1[j] -= -(i+1)*strength*coefs[i]*M_PI*std::cos(2*(i+1)*M_PI*(j+1)*h)/h;
    		plist2[j] += (i+1)*ratio*strength*coefs[i]*M_PI*std::cos(2*(i+1)*M_PI*j*h)/h;
    		qlist2[j] -= (i+1)*ratio*strength*coefs[i]*M_PI*std::cos(2*(i+1)*M_PI*(j+1)*h)/h;
 			}
 		// printfln("%d %f %f %f %f",j,plist1[j],qlist1[j],plist2[j],qlist2[j]);
		}
	auto ampo1m = AutoMPO(sites), ampo1p = AutoMPO(sites), ampo2m = AutoMPO(sites), ampo2p = AutoMPO(sites);
	for(auto j : range1(bins))
		{
		ampo1m += plist1[j-1]*std::exp(-dz),"S+",j,"S-",j%bins+1;
		ampo1m += -plist1[j-1],"projUp",j,"projDn",j%bins+1;
		ampo1m += qlist1[j-1]*std::exp(dz),"S-",j,"S+",j%bins+1;
		ampo1m += -qlist1[j-1],"projDn",j,"projUp",j%bins+1;
		ampo1p += plist1[j-1]*std::exp(dz),"S+",j,"S-",j%bins+1;
		ampo1p += -plist1[j-1],"projUp",j,"projDn",j%bins+1;
		ampo1p += qlist1[j-1]*std::exp(-dz),"S-",j,"S+",j%bins+1;
		ampo1p += -qlist1[j-1],"projDn",j,"projUp",j%bins+1;
		ampo2m += plist2[j-1]*std::exp(-dz),"S+",j,"S-",j%bins+1;
		ampo2m += -plist2[j-1],"projUp",j,"projDn",j%bins+1;
		ampo2m += qlist2[j-1]*std::exp(dz),"S-",j,"S+",j%bins+1;
		ampo2m += -qlist2[j-1],"projDn",j,"projUp",j%bins+1;
		ampo2p += plist2[j-1]*std::exp(dz),"S+",j,"S-",j%bins+1;
		ampo2p += -plist2[j-1],"projUp",j,"projDn",j%bins+1;
		ampo2p += qlist2[j-1]*std::exp(-dz),"S-",j,"S+",j%bins+1;
		ampo2p += -qlist2[j-1],"projDn",j,"projUp",j%bins+1;
		}
	auto W1m = toMPO(ampo1m), W1p = toMPO(ampo1p), W2m = toMPO(ampo2m), W2p = toMPO(ampo2p);

	// auto H = W1m;
	// auto Hfull = H(1)*H(2)*H(3)*H(4);
	// auto inds = Hfull.inds();
	// auto C = std::get<0>(combiner(inds[0],inds[2],inds[4],inds[6]));
	// auto Cp = std::get<0>(combiner(inds[1],inds[3],inds[5],inds[7]));
	// auto Hfullmat = C*Hfull*Cp;
	// PrintData(Hfullmat);

	auto anop = AutoMPO(sites);
	for(auto j : range1(bins))
		{
		anop += "Sz",j;
		}
  	auto Nop = toMPO(anop);
	int spin = inner(psi0,Nop,psi0);
	printfln("Driving frequency: %f",freq);
	printfln("\nParticle number: %d",spin+bins/2);

	// auto expH1m = toExpH(ampo1m,1/(2*freq)), expH1p = toExpH(ampo1p,1/(2*freq)), expH2m = toExpH(ampo2m,1/(2*freq)), expH2p = toExpH(ampo2p,1/(2*freq));
	// auto Cm = nmultMPO(prime(expH1m),expH2m,{"MaxDim",500,"Cutoff",1E-13}), Cp = nmultMPO(prime(expH1p),expH2p,{"MaxDim",500,"Cutoff",1E-13});
	// PrintData(expH1m);
	// PrintData(expH2m);
	// PrintData(Cm);

	auto sweeps = Sweeps(30);
	sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
	sweeps.cutoff() = 1E-13;
	sweeps.niter() = 10;
	sweeps.noise() = 0.0;
	sweeps.alpha() = 0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002;
	println();
	println(sweeps);

	println("\nStart DMRG");

	auto psim = std::get<1>(tree_dmrg(W2m,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"}));

	// auto psi2mfull = psi2m(0)*psi2m(1)*psi2m(2)*psi2m(3)*psi2m(4)*psi2m(5)*psi2m(6)*psi2m(7)*psi2m(8)*psi2m(9)*psi2m(10)*psi2m(11)*psi2m(12)*psi2m(13)*psi2m(14);
	// auto inds2m = psi2mfull.inds();
	// auto C2m = std::get<0>(combiner(inds2m[0],inds2m[1],inds2m[2],inds2m[3],inds2m[4],inds2m[5],inds2m[6],inds2m[7],inds2m[8],inds2m[9],inds2m[10],inds2m[11],inds2m[12],inds2m[13],inds2m[14],inds2m[15]));
	// auto psi2mfullmat = C2m*psi2mfull;
	// PrintData(psi2mfullmat);

	auto psip = std::get<1>(tree_dmrg(W2p,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"}));

	// auto psi2pfull = psi2p(0)*psi2p(1)*psi2p(2)*psi2p(3)*psi2p(4)*psi2p(5)*psi2p(6)*psi2p(7)*psi2p(8)*psi2p(9)*psi2p(10)*psi2p(11)*psi2p(12)*psi2p(13)*psi2p(14);
	// auto inds2p = psi2pfull.inds();
	// auto C2p = std::get<0>(combiner(inds2p[0],inds2p[1],inds2p[2],inds2p[3],inds2p[4],inds2p[5],inds2p[6],inds2p[7],inds2p[8],inds2p[9],inds2p[10],inds2p[11],inds2p[12],inds2p[13],inds2p[14],inds2p[15]));
	// auto psi2pfullmat = C2p*psi2pfull;
	// PrintData(psi2pfullmat);

	// auto [energym,psim] = tree_dmrg(Cm,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"});

	auto period = 1/freq;
	auto deltat = period/nstages;

	auto sweeps1 = Sweeps(1);
	sweeps1.maxdim() = 300;
	sweeps1.cutoff() = 1E-13;
	sweeps1.niter() = 100;
	sweeps1.noise() = 0.0;
	println(sweeps1);

	for(auto i : range1(10))
		{
		auto psim0 = psim;
		auto psip0 = psip;
		for(Real t = 0.0; t < period; t += deltat)
			{
			if(t < period/2)
				{
				psim = std::get<1>(tree_tdvp(W1m,psim,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
				psip = std::get<1>(tree_tdvp(W1p,psip,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
				}
			else
				{
				psim = std::get<1>(tree_tdvp(W2m,psim,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
				psip = std::get<1>(tree_tdvp(W2p,psip,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
				}
			}
		printfln("\n%d: exp(v-) = %f, exp(v+) = %f",i,inner(psim0,psim),inner(psip0,psip));
		psim.normalize();
		psip.normalize();
		}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printfln("\nElapsed time: %f s",elapsed.count());
	}