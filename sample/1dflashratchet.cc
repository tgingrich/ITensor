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
	Real len = 0.0, strength = 0.0, ratio = 0.0, di = 0.0;
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
				di = std::stod(val);;
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
	for(auto i : range(bins)) // Note: sites are labelled from 1
		{
		if(i%(int)ceil((Real)bins/nparticles)==0) state.set(i+1,"Up");
		else state.set(i+1,"Dn");
		}
	auto psi0 = BinaryTree(state);

	Real dz = 0.001;
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
 		printfln("%d %f %f %f %f",j,plist1[j],qlist1[j],plist2[j],qlist2[j]);
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
	// auto W1m = toMPO(ampo1m)

	auto anop = AutoMPO(sites);
	for(auto j : range1(bins))
		{
		anop += "Sz",j;
		}
  	auto Nop = toMPO(anop);
	int spin = inner(psi0,Nop,psi0);
	printfln("\nDriving frequency: %f",freq);
	printfln("\nParticle number: %d",spin+bins/2);

	// auto H = W1m;
	// auto Hfull = H(1)*H(2)*H(3)*H(4);
	// auto inds = Hfull.inds();
	// auto C = std::get<0>(combiner(inds[0],inds[2],inds[4],inds[6]));
	// auto Cp = std::get<0>(combiner(inds[1],inds[3],inds[5],inds[7]));
	// auto Hfullmat = C*Hfull*Cp;
	// PrintData(Hfullmat);

	// auto sweeps = Sweeps(8);
	// sweeps.maxdim() = 16,16,16,16,16,16,16,16;
	// // sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
	// sweeps.cutoff() = 1E-13;
	// sweeps.niter() = 10;
	// sweeps.noise() = 0.0;
	// sweeps.alpha() = 0.1,0.1,0.05,0.05,0.02,0.02,0.01,0.01;
	// // sweeps.alpha() = 0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002;
	// println();
	// println(sweeps);

	// println("\nStart DMRG");

	// auto [energym,psim] = tree_dmrg(W1m,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"});
	// auto [energyp,psip] = tree_dmrg(W1p,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"});

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printfln("\nElapsed time: %f s",elapsed.count());
	}