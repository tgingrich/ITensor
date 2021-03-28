#include <chrono>
#include <fstream>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(int argc, char** argv)
	{
	auto start = std::chrono::high_resolution_clock::now();
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
	ifs.open("freqs.in");
	std::vector<Real> freqs;
	while(ifs >> val)
		{
		freqs.push_back(std::stod(val));
		}
	ifs.close();
	bool rev = false;
	if(freqs.back()<0)
		{
		rev = true;
		freqs.pop_back();
		std::reverse(freqs.begin(),freqs.end());
		}
	ifs.open("coefs.in");
	std::vector<Real> coefs;
	while(ifs >> val)
		{
		coefs.push_back(std::stod(val));
		}
	ifs.close();
	int maxdim = 300, se1 = 1, se2 = 0, tdvp_freq = 1E5;
	Real co1 = 1.0E-13, co2 = 1.0E-13;
	if (argc > 3) maxdim = std::stoi(argv[3]);
	if (argc > 4) se1 = std::stoi(argv[4]);
	if (argc > 5) se2 = std::stoi(argv[5]);
	if (argc > 6) co1 = std::stod(argv[6]);
	if (argc > 7) co2 = std::stod(argv[7]);
	if (argc > 8) tdvp_freq = std::stoi(argv[8]);
	printfln("%d %d %d %.10e %.10e %d",maxdim,se1,se2,co1,co2,tdvp_freq);
	println();

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
	auto ampoavg = AutoMPO(sites);
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
		ampoavg += (plist1[j-1]+plist2[j-1])/2,"S+",j,"S-",j%bins+1;
		ampoavg += -(plist1[j-1]+plist2[j-1])/2,"projDn",j,"projUp",j%bins+1;
		ampoavg += (qlist1[j-1]+qlist2[j-1])/2,"S-",j,"S+",j%bins+1;
		ampoavg += -(qlist1[j-1]+qlist2[j-1])/2,"projUp",j,"projDn",j%bins+1;
		}
	auto W1m = toMPO(ampo1m), W1p = toMPO(ampo1p), W2m = toMPO(ampo2m), W2p = toMPO(ampo2p);
	auto W1 = toMPO(ampo1), W2 = toMPO(ampo2);
	auto Wavg = toMPO(ampoavg);

	auto anop = AutoMPO(sites);
	for(auto j : range1(bins))
		{
		anop += "Sz",j;
		}
  	auto Nop = toMPO(anop);
	int spin = inner(psi0,Nop,psi0);
	printfln("Particle number: %d",bins/2-spin);
	println();

	println("Start DMRG");
	println();

	auto sweeps0 = Sweeps(1);
	sweeps0.maxdim() = maxdim;
	sweeps0.cutoff() = co1;
	sweeps0.niter() = 100;
	sweeps0.noise() = 0.0;

	auto sweeps = Sweeps(30);
	sweeps.maxdim() = maxdim;
	sweeps.cutoff() = co1;
	sweeps.niter() = 100;
	sweeps.noise() = 0.0;
	println(sweeps);
	println();

	if(rev)
		{
		psi0 = std::get<1>(tree_dmrg(Wavg,psi0,sweeps0,{"NumCenter",1,"WhichEig","LargestReal","Quiet",}));
		psi0 = std::get<1>(tree_dmrg(Wavg,psi0,sweeps,{"NumCenter",1,"WhichEig","LargestReal","SubspaceExpansion",se1==1,"Quiet",}));
		}
	else
		{
		psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps0,{"NumCenter",1,"WhichEig","LargestReal","Quiet",}));
		psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps,{"NumCenter",1,"WhichEig","LargestReal","SubspaceExpansion",se1==1,"Quiet",}));
		}

	println("Start TDVP");
	println();
	for(Real freq : freqs)
		{
		printfln("Driving frequency: %f",freq);
		println();
		auto nstages = std::max(10,(int)(tdvp_freq/freq));

		auto sweeps1 = Sweeps(nstages/2);
		sweeps1.maxdim() = maxdim;
		sweeps1.cutoff() = co2;
		sweeps1.niter() = 100;
		sweeps1.noise() = 0.0;
		println(sweeps1);
		println();

		auto period = 1/freq;
		auto deltat = period/nstages;
		int maxiter = 10*freq;
		Real thresh = 0.1;
		Real mean, var;
		int iter = 0;
		auto psim = psi0;
		auto psip = psi0;
		while(iter<maxiter)
			{
			auto psim0 = psim;
			auto psip0 = psip;
			auto mean0 = mean;
			psim = std::get<1>(tree_tdvp(W1m,psim,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
			psim = std::get<1>(tree_tdvp(W2m,psim,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
			psip = std::get<1>(tree_tdvp(W1p,psip,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
			psip = std::get<1>(tree_tdvp(W2p,psip,deltat,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
			auto left = std::log(inner(psim0,psim))/period, right = std::log(inner(psip0,psip))/period;
			mean = (right-left)/(2*dz);
			var = (right+left)/(dz*dz);
			printfln("%d: v- = %f, v+ = %f, j = %f, dj = %f",iter++,left,right,mean,mean-mean0);
			println();
			psim.normalize();
			psip.normalize();
			if(dens>0)
				{
				psi0 = std::get<1>(tree_tdvp(W1,psi0,deltat,sweeps1,{"NumCenter",1,"SubspaceExpansion",se2==1,"Quiet",}));
				psi0 = std::get<1>(tree_tdvp(W2,psi0,deltat,sweeps1,{"NumCenter",1,"SubspaceExpansion",se2==1,"Quiet",}));
				}
			if(fabs(mean-mean0)<thresh && fabs(left)<0 && fabs(right)<0) break;
			}
		if(iter==maxiter)
			{
			println("Max iterations reached!");
			println();
			}
		printfln("{jbar, varj} = {%f, %f}",mean,var);
		println();

		if(dens>0)
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
		}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printfln("Elapsed time: %f s",elapsed.count());
	}