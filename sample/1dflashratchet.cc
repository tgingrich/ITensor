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
				di = std::stod(val);
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

	// auto H = W2p;
	// auto Hfull = H(1)*H(2)*H(3)*H(4)*H(5)*H(6)*H(7)*H(8);
	// auto inds = Hfull.inds();
	// auto C = std::get<0>(combiner(inds[0],inds[2],inds[4],inds[6],inds[8],inds[10],inds[12],inds[14]));
	// auto Cp = std::get<0>(combiner(inds[1],inds[3],inds[5],inds[7],inds[9],inds[11],inds[13],inds[15]));
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
	auto psip = std::get<1>(tree_dmrg(W2p,psi0,sweeps,{"NumCenter",2,"Order","PostOrder","Quiet",true,"WhichEig","LargestReal"}));
	// auto psim = psi0, psip = psi0;

	int nstages = std::max(100,(int)(10000/freq));
	auto period = 1/freq;
	auto deltat = period/nstages;

	auto sweeps1 = Sweeps(nstages/2);
	sweeps1.maxdim() = 300;
	sweeps1.cutoff() = 1E-13;
	sweeps1.niter() = 100;
	sweeps1.noise() = 0.0;
	sweeps1.alpha() = 0.001;
	println();
	println(sweeps1);

	println("\nStart TDVP");

	int maxiter = 10*freq;
	Real thresh = 1.0E-4;
	Real mean, var;
	int iter = 0;
	while(iter<maxiter)
		{
		auto psim0 = psim;
		auto psip0 = psip;
		auto mean0 = mean;
		psim = std::get<1>(tree_tdvp(W1m,psim,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
		psim = std::get<1>(tree_tdvp(W2m,psim,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
		psip = std::get<1>(tree_tdvp(W1p,psip,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
		psip = std::get<1>(tree_tdvp(W2p,psip,deltat,sweeps1,{"NumCenter",1,"Order","PostOrder","DoNormalize",false,"Quiet",}));
		auto left = std::log(inner(psim0,psim))/period, right = std::log(inner(psip0,psip))/period;
		printfln("\n%d: v- = %f, v+ = %f",iter++,left,right);
		mean = (right-left)/(2*dz);
		var = (right+left)/(dz*dz);
		psim.normalize();
		psip.normalize();
		// printfln("%f",fabs(mean-mean0));
		if(fabs(mean-mean0)<thresh) break;
		}
	if(iter==maxiter) println("\nMax iterations reached!");
	printfln("\n{jbar, varj} = {%f, %f}",mean,var);
	// auto sweeps = Sweeps(nstages/2);
	// sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300;
	// sweeps.cutoff() = 1E-13;
	// sweeps.niter() = 10;
	// sweeps.noise() = 0.0;
	// sweeps.alpha() = 0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002;
	// for(auto i : range1(50))
	// 	{
	// 	auto psim0 = psim;
	// 	auto psip0 = psip;
	// 	psim = std::get<1>(tree_tdvp(W1m,psim,deltat,i==1?sweeps:sweeps1,{"NumCenter",2,"Order","PostOrder","DoNormalize",false,"Quiet",}));
	// 	psim = std::get<1>(tree_tdvp(W2m,psim,deltat,sweeps1,{"NumCenter",2,"Order","PostOrder","DoNormalize",false,"Quiet",}));
	// 	psip = std::get<1>(tree_tdvp(W1p,psip,deltat,i==1?sweeps:sweeps1,{"NumCenter",2,"Order","PostOrder","DoNormalize",false,"Quiet",}));
	// 	psip = std::get<1>(tree_tdvp(W2p,psip,deltat,sweeps1,{"NumCenter",2,"Order","PostOrder","DoNormalize",false,"Quiet",}));
	// 	printfln("\n%d: v- = %f, v+ = %f",i,std::log(inner(psim0,psim))/period,std::log(inner(psip0,psip))/period);
	// 	psim.normalize();
	// 	psip.normalize();
	// 	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printfln("\nElapsed time: %f s",elapsed.count());

	}