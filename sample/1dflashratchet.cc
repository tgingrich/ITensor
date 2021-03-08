#include <chrono>
#include <fstream>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(int argc, char** argv)
	{
	auto start = std::chrono::high_resolution_clock::now();
	auto freq = atof(argv[3]);
	int nparticles = atoi(argv[2]);
	std::ifstream ifs;
	ifs.open(argv[1]);
	std::string type, val;
	Real len = 0.0, strength = 0.0, ratio = 0.0, di = 0.0;
	int bins = 0;
	int idx = 0;
	int dens = 0;
	while(ifs >> type >> val)
		{
		switch(idx++)
			{
			case 0: len = std::stod(val); break;
			case 1: bins = std::stoi(val); break;
			case 2: strength = std::stod(val); break;
			case 3: ratio = std::stod(val); break;
			case 4: di = std::stod(val); break;
			case 5: dens = std::stoi(val);
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
	int maxdim = 300, nstages = std::max(10,(int)(1E5/freq));
	if (argc > 4)
		{
	    maxdim = std::stoi(argv[4]);
	    nstages = std::max(10,(int)(std::stoi(argv[5])/freq));
		}

	auto sites = SpinHalf(bins,{"ConserveQNs",true});
	auto state = InitState(sites);
	int count = 0;
	for(auto i : range(bins))
		{
		if(i%(bins/nparticles)==0 && count++ < nparticles) state.set(i+1,"Dn");
		else state.set(i+1,"Up");
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
		}
	auto ampo1m = AutoMPO(sites), ampo1p = AutoMPO(sites), ampo2m = AutoMPO(sites), ampo2p = AutoMPO(sites);
	auto ampo1 = AutoMPO(sites), ampo2 = AutoMPO(sites);
	for(auto j : range1(bins))
		{
		ampo1m += plist1[j-1]*std::exp(-dz),"S+",j,"S-",j%bins+1;
		ampo1m += -plist1[j-1],"projDn",j,"projUp",j%bins+1;
		ampo1m += qlist1[j-1]*std::exp(dz),"S-",j,"S+",j%bins+1;
		ampo1m += -qlist1[j-1],"projUp",j,"projDn",j%bins+1;
		ampo1p += plist1[j-1]*std::exp(dz),"S+",j,"S-",j%bins+1;
		ampo1p += -plist1[j-1],"projDn",j,"projUp",j%bins+1;
		ampo1p += qlist1[j-1]*std::exp(-dz),"S-",j,"S+",j%bins+1;
		ampo1p += -qlist1[j-1],"projUp",j,"projDn",j%bins+1;
		ampo2m += plist2[j-1]*std::exp(-dz),"S+",j,"S-",j%bins+1;
		ampo2m += -plist2[j-1],"projDn",j,"projUp",j%bins+1;
		ampo2m += qlist2[j-1]*std::exp(dz),"S-",j,"S+",j%bins+1;
		ampo2m += -qlist2[j-1],"projUp",j,"projDn",j%bins+1;
		ampo2p += plist2[j-1]*std::exp(dz),"S+",j,"S-",j%bins+1;
		ampo2p += -plist2[j-1],"projDn",j,"projUp",j%bins+1;
		ampo2p += qlist2[j-1]*std::exp(-dz),"S-",j,"S+",j%bins+1;
		ampo2p += -qlist2[j-1],"projUp",j,"projDn",j%bins+1;
		ampo1 += plist1[j-1],"S+",j,"S-",j%bins+1;
		ampo1 += -plist1[j-1],"projDn",j,"projUp",j%bins+1;
		ampo1 += qlist1[j-1],"S-",j,"S+",j%bins+1;
		ampo1 += -qlist1[j-1],"projUp",j,"projDn",j%bins+1;
		ampo2 += plist2[j-1],"S+",j,"S-",j%bins+1;
		ampo2 += -plist2[j-1],"projDn",j,"projUp",j%bins+1;
		ampo2 += qlist2[j-1],"S-",j,"S+",j%bins+1;
		ampo2 += -qlist2[j-1],"projUp",j,"projDn",j%bins+1;
		}
	auto W1m = toMPO(ampo1m), W1p = toMPO(ampo1p), W2m = toMPO(ampo2m), W2p = toMPO(ampo2p);
	auto W1 = toMPO(ampo1), W2 = toMPO(ampo2);

	auto anop = AutoMPO(sites);
	for(auto j : range1(bins))
		{
		anop += "Sz",j;
		}
  	auto Nop = toMPO(anop);
	int spin = inner(psi0,Nop,psi0);
	printfln("Driving frequency: %f",freq);
	printfln("\nParticle number: %d",bins/2-spin);

	// auto sweeps = Sweeps(40);
	// if(maxdim < 150) sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,maxdim;
	// else if(maxdim < 200) sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,maxdim;
	// else if(maxdim < 250) sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,maxdim;
	// else if(maxdim < 300) sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,maxdim;
	// else sweeps.maxdim() = 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,maxdim;
	// sweeps.cutoff() = 1E-13;
	// sweeps.niter() = 10;
	// sweeps.noise() = 0.0;
	// sweeps.alpha() = 0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.02,0.02,0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.002,0.002,0.002,0.002,0.002;
	// println();
	// println(sweeps);

	// auto sweeps0 = Sweeps(1);
	// sweeps0.maxdim() = maxdim;
	// sweeps0.cutoff() = 1E-13;
	// sweeps0.niter() = 100;
	// sweeps0.noise() = 0.0;
	// sweeps0.alpha() = 0.001;

	// auto sweeps = Sweeps(30);
	// sweeps.maxdim() = maxdim;
	// sweeps.cutoff() = 1E-13;
	// sweeps.niter() = 100;
	// sweeps.noise() = 0.0;
	// sweeps.alpha() = 0.001;

	// auto sweeps0 = Sweeps(10);
	// sweeps0.maxdim() = maxdim;
	// sweeps0.cutoff() = 1E-13;
	// sweeps0.niter() = 100;
	// sweeps0.noise() = 0.0;
	// sweeps0.alpha() = 0.001;

	auto sweeps = Sweeps(30);
	if(maxdim < 150) sweeps.maxdim() = maxdim;
	else if(maxdim < 200) sweeps.maxdim() = 100,110,120,130,140,150,maxdim;
	else if(maxdim < 250) sweeps.maxdim() = 100,110,120,130,140,150,160,170,180,190,200,maxdim;
	else if(maxdim < 300) sweeps.maxdim() = 100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,maxdim;
	else sweeps.maxdim() = 100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,maxdim;
	sweeps.cutoff() = 1E-13;
	sweeps.niter() = 100;
	sweeps.noise() = 0.0;
	sweeps.alpha() = 0.001;

	auto sweeps0 = Sweeps(10);
	sweeps0.maxdim() = 10,20,30,40,50,60,70,80,90,100;
	sweeps0.cutoff() = 1E-13;
	sweeps0.niter() = 100;
	sweeps0.noise() = 0.0;
	sweeps0.alpha() = 0.1,0.05,0.02,0.01,0.005,0.005,0.002,0.002,0.001,0.001;

	println("\nStart DMRG");

	// auto datam = tree_dmrg(W2m,psi0,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",});
	// auto psim = std::get<1>(datam);
	// while(fabs(std::get<0>(datam)) > 1)
	// 	{
	// 	datam = tree_dmrg(W2m,psim,sweeps0,{"NumCenter",2,"WhichEig","LargestReal","Quiet",});
	// 	psim = std::get<1>(datam);
	// 	}
	// auto datap = tree_dmrg(W2p,psi0,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",});
	// auto psip = std::get<1>(datap);
	// while(fabs(std::get<0>(datap)) > 1)
	// 	{
	// 	datap = tree_dmrg(W2p,psip,sweeps0,{"NumCenter",2,"WhichEig","LargestReal","Quiet",});
	// 	psip = std::get<1>(datap);
	// 	}

	// if(dens > 0)
	// 	{
	// 	auto data0 = tree_dmrg(W2,psi0,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",});
	// 	psi0 = std::get<1>(data0);
	// 	while(fabs(std::get<0>(data0)) > 1)
	// 		{
	// 		data0 = tree_dmrg(W2,psi0,sweeps0,{"NumCenter",2,"WhichEig","LargestReal","Quiet",});
	// 		psi0 = std::get<1>(data0);
	// 		}
	// 	}

	// auto psim = psi0, psip = psi0;
	// int duplicates = 10;
	// auto sweeps = Sweeps(duplicates);
	// for(auto j : range1(maxdim/10))
	// 	{
	// 	sweeps.maxdim() = 10*j;
	// 	sweeps.cutoff() = 1E-13;
	// 	sweeps.niter() = 10;
	// 	sweeps.noise() = 0.0;
	// 	sweeps.alpha() = 0.1*std::exp(-0.2*j);
	// 	psim = std::get<1>(tree_dmrg(W2m,psim,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",}));
	// 	psip = std::get<1>(tree_dmrg(W2p,psip,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",}));
	// 	if(dens > 0) psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",}));
	// 	}

	auto psim = std::get<1>(tree_dmrg(W2m,psi0,sweeps0,{"NumCenter",2,"WhichEig","LargestReal","Quiet",}));
	psim = std::get<1>(tree_dmrg(W2m,psim,sweeps,{"NumCenter",2,"WhichEig","LargestReal","SubspaceExpansion",false,"Quiet",}));
	auto psip = std::get<1>(tree_dmrg(W2p,psi0,sweeps0,{"NumCenter",2,"WhichEig","LargestReal","Quiet",}));
	psip = std::get<1>(tree_dmrg(W2p,psip,sweeps,{"NumCenter",2,"WhichEig","LargestReal","SubspaceExpansion",false,"Quiet",}));
	if(dens > 0)
		{
		psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps,{"NumCenter",2,"WhichEig","LargestReal","Quiet",}));
		psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps0,{"NumCenter",2,"WhichEig","LargestReal","SubspaceExpansion",false,"Quiet",}));
		}

	auto period = 1/freq;
	auto deltat = period/nstages;
	// auto deltat = period/100;

	auto sweeps1 = Sweeps(nstages/2);
	// auto sweeps1 = Sweeps(1);
	sweeps1.maxdim() = maxdim;
	sweeps1.cutoff() = 1E-13;
	sweeps1.niter() = 100;
	sweeps1.noise() = 0.0;
	sweeps1.alpha() = 0.001;
	println();
	println(sweeps1);

	println("\nStart TDVP");

	// auto psim = std::get<1>(tree_tdvp(W2m,psi0,deltat,sweeps,{"NumCenter",2,"Quiet",}));
	// auto psip = std::get<1>(tree_tdvp(W2p,psi0,deltat,sweeps,{"NumCenter",2,"Quiet",}));

	// if(dens > 0) psi0 = std::get<1>(tree_tdvp(W2,psi0,deltat,sweeps,{"NumCenter",2,"Quiet",}));

	// PrintData(psim(0).inds());
	// PrintData(psim(1).inds());
	// PrintData(psim(2).inds());
	// PrintData(psim(3).inds());
	// PrintData(psim(4).inds());
	// PrintData(psim(5).inds());
	// PrintData(psim(6).inds());
	// for (int i = 0; i < 10; ++i)
	// 	{
	// 	printfln("%f, %f", inner(psim,psim), inner(psim,W1m,psim));
	// 	auto Hfull = psim(0) * psim(1) * psim(2) * psim(3) * psim(4) * psim(5) * psim(6);
	// 	auto inds = Hfull.inds();
	// 	auto C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7]));
	// 	auto psi1fullmat = C * Hfull;
	// 	PrintData(psi1fullmat);
	// 	psim = std::get<1>(tree_tdvp(W1m,psim,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"Quiet",}));
	// 	}

	int maxiter = 10*freq;
	Real thresh = 1.0E-2;
	Real mean, var;
	int iter = 0;
	while(iter<maxiter)
		{
		for(auto j : range(bins-1))
			{
			println(j);
			PrintData(psim(j).inds());
			PrintData(psip(j).inds());
			}
		auto psim0 = psim;
		auto psip0 = psip;
		auto mean0 = mean;
		psim = std::get<1>(tree_tdvp(W1m,psim,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"Quiet",}));
		psim = std::get<1>(tree_tdvp(W2m,psim,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"Quiet",}));
		psip = std::get<1>(tree_tdvp(W1p,psip,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"Quiet",}));
		psip = std::get<1>(tree_tdvp(W2p,psip,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"Quiet",}));
		auto left = std::log(inner(psim0,psim))/period, right = std::log(inner(psip0,psip))/period;
		mean = (right-left)/(2*dz);
		var = (right+left)/(dz*dz);
		printfln("\n%d: v- = %f, v+ = %f, j = %f, dj = %f",iter++,left,right,mean,mean-mean0);
		psim.normalize();
		psip.normalize();
		if(dens > 0)
			{
			psi0 = std::get<1>(tree_tdvp(W1,psi0,deltat,sweeps1,{"NumCenter",1,"Quiet",}));
			psi0 = std::get<1>(tree_tdvp(W2,psi0,deltat,sweeps1,{"NumCenter",1,"Quiet",}));
			}
		if(fabs(mean-mean0)<thresh) break;
		}
	if(iter==maxiter) println("\nMax iterations reached!");
	printfln("\n{jbar, varj} = {%f, %f}",mean,var);

	if(dens > 0)
		{
		auto sweeps2 = Sweeps(nstages/dens);
		sweeps2.maxdim() = maxdim;
		sweeps2.cutoff() = 1E-13;
		sweeps2.niter() = 100;
		sweeps2.noise() = 0.0;
		sweeps2.alpha() = 0.001;
		std::ofstream ofs;
		ofs.open("dens_"+std::to_string(nparticles)+"_"+std::to_string((int)freq)+"_"+std::to_string(bins)+".txt");
		for(auto j : range(dens))
			{
			psi0 = std::get<1>(tree_tdvp(j*2/dens == 0 ? W1 : W2,psi0,deltat,sweeps1,{"NumCenter",1,"Quiet",}));
			for(auto n : range1(bins))
				{
				ofs << siteval(psi0,n)[1] << " ";
				}
			ofs << std::endl;
			}
		ofs.close();
		}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printfln("\nElapsed time: %f s",elapsed.count());

	}