#include <chrono>
#include <fstream>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

long long choose(int n, int k)
	{
	if(k == 0) return 1;
	return (n*choose(n-1,k-1))/k;
	}

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
	int dbl = 0;
	while(ifs >> type >> val)
		{
		switch(idx++)
			{
			case 0: len = std::stod(val); break;
			case 1: bins = std::stoi(val); break;
			case 2: strength = std::stod(val); break;
			case 3: ratio = std::stod(val); break;
			case 4: di = std::stod(val); break;
			case 5: dens = std::stoi(val); break;
			case 6: dbl = std::stoi(val);
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
	int maxdim = 300, se1 = 1, se2 = 0, tdvp_freq1 = 1E5, tdvp_freq2 = 1E5, anneal = 0;
	Real co1 = 1.0E-13, co2 = 1.0E-13, alpha = 0.1;
	if (argc > 4) maxdim = std::stoi(argv[4]);
	if (argc > 5) se1 = std::stoi(argv[5]);
	if (argc > 6) se2 = std::stoi(argv[6]);
	if (argc > 7) co1 = std::stod(argv[7]);
	if (argc > 8) co2 = std::stod(argv[8]);
	if (argc > 9) tdvp_freq1 = std::stoi(argv[9]);
	if (argc > 10) tdvp_freq2 = std::stoi(argv[10]);
	if (argc > 11) alpha = std::stod(argv[11]);
	if (argc > 12) anneal = std::stoi(argv[12]);
	printfln("%d %d %d %.10e %.10e %d %d %f %d",maxdim,se1,se2,co1,co2,tdvp_freq1,tdvp_freq2,alpha,anneal);
	printfln("\nDriving frequency: %f",freq);

	bins /= pow2(dbl);
	Real dz = 0.0001;
	BinaryTree psi0;
	MPO W1m, W1p, W2m, W2p, W1, W2;
	std::vector<BinaryTree> smalltreelist, largetreelist;
	int minnp = dbl ? 0 : nparticles, maxnp = dbl ? std::min(nparticles,bins) : nparticles;
	auto sites = SpinHalf(bins,{"ConserveQNs",true});
	auto state = InitState(sites);
	for(auto np : range1(minnp,maxnp))
		{
		int count = 0;
		for(auto i : range(bins))
			{
			if(np && i%(bins/np)==0 && count++ < np) state.set(i+1,"Dn");
			else state.set(i+1,"Up");
			}
		psi0 = BinaryTree(state);
		largetreelist.push_back(psi0);
		}
	for(auto level : range(dbl+1))
		{
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
		W1m = toMPO(ampo1m);
		W1p = toMPO(ampo1p);
		W2m = toMPO(ampo2m);
		W2p = toMPO(ampo2p);
		W1 = toMPO(ampo1);
		W2 = toMPO(ampo2);
		minnp = level<dbl ? 0 : nparticles;
		maxnp = level<dbl ? std::min(nparticles,bins) : nparticles;
		int idx=0;
		for(auto np : range1(minnp,maxnp))
			{
			psi0 = largetreelist[idx];
			auto anop = AutoMPO(sites);
			for(auto j : range1(bins))
				{
				anop += "Sz",j;
				}
		  	auto Nop = toMPO(anop);
			int spin = inner(psi0,Nop,psi0);
			printfln("\nBin number: %d",bins);
			printfln("Particle number: %d",bins/2-spin);

			// if(bins==8)
			// 	{
			// 	auto Hfull = psi0(0) * psi0(1) * psi0(2) * psi0(3) * psi0(4) * psi0(5) * psi0(6)/* * psi0(7) * psi0(8) * psi0(9) * psi0(10) * psi0(11) * psi0(12) * psi0(13) * psi0(14)*/;
			// 	auto inds = Hfull.inds();
			// 	auto C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7]/*, inds[8], inds[9], inds[10], inds[11], inds[12], inds[13], inds[14], inds[15]*/));
			// 	auto psi1fullmat = C * Hfull;
			// 	PrintData(psi1fullmat);
			// 	}

			println();
			for(auto n : range1(bins)) printf("%f ",siteval(psi0,n)[1]);
			println();
			if(np!=0 && np!=bins)
				{
				printfln("\nInitial energy: %f",inner(psi0,W2,psi0));

				for(auto j : range(psi0.size()))
					{
					println(j);
					PrintData(psi0(j).inds());
					}

				auto sweeps0 = Sweeps(1);
				sweeps0.maxdim() = maxdim;
				sweeps0.cutoff() = co1;
				sweeps0.niter() = 100;
				sweeps0.noise() = 0.0;
				sweeps0.alpha() = alpha;

				println("\nStart DMRG");
				println();

				if(anneal)
					{
					auto sweeps = Sweeps(anneal);
					for(auto j : range1(maxdim/10))
						{
						sweeps.maxdim() = 10*j;
						sweeps.cutoff() = 1E-15;
						sweeps.niter() = 100;
						sweeps.noise() = 0.0;
						sweeps.alpha() = alpha<0 ? std::exp(-0.2*j) : alpha;
						println(sweeps);
						psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps,{"NumCenter",1,"WhichEig","LargestReal","Quiet",}));
						}
					}
				else
					{
					auto sweeps = Sweeps(30);
					sweeps.maxdim() = maxdim;
					sweeps.cutoff() = co1;
					sweeps.niter() = 100;
					sweeps.noise() = 0.0;
					sweeps.alpha() = alpha;
					println(sweeps);
					psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps0,{"NumCenter",1,"WhichEig","LargestReal","Quiet",}));
					psi0 = std::get<1>(tree_dmrg(W2,psi0,sweeps,{"NumCenter",1,"WhichEig","LargestReal","SubspaceExpansion",se1==1,"Quiet",}));
					}
				println();
				for(auto n : range1(bins)) printf("%f ",siteval(psi0,n)[1]);
				println();
			
				for(auto j : range(psi0.size()))
					{
					println(j);
					PrintData(psi0(j).inds());
					}
				}
			else psi0.orthogonalize();

			// if(bins==16)
			// 	{
			// 	auto Hfull = psi0(0) * psi0(1) * psi0(2) * psi0(3) * psi0(4) * psi0(5) * psi0(6) * psi0(7) * psi0(8) * psi0(9) * psi0(10) * psi0(11) * psi0(12) * psi0(13) * psi0(14);
			// 	auto inds = Hfull.inds();
			// 	auto C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7], inds[8], inds[9], inds[10], inds[11], inds[12], inds[13], inds[14], inds[15]));
			// 	auto psi1fullmat = C * Hfull;
			// 	PrintData(psi1fullmat);
			// 	}
			// PrintData(psi0);

			smalltreelist.push_back(psi0);
			++idx;
			}

		if(level<dbl)
			{
			bins *= 2;
			sites = SpinHalf(bins,{"ConserveQNs",true});
			state = InitState(sites);
			// psi0 = doubleTree(psi0,state);
			minnp = level==dbl-1 ? nparticles : 0;
			maxnp = level==dbl-1 ? nparticles : std::min(nparticles,bins);
			largetreelist.clear();
			for(auto np : range1(minnp,maxnp))
				{
				BinaryTree largetree;
				std::vector<Real> plist2(bins), qlist2(bins);
				for(auto j : range(bins))
					{
					plist2[j] = qlist2[j] = di/std::pow(h,2);
					for (auto i : range((int)coefs.size()))
						{
			    		plist2[j] += (i+1)*ratio*strength*coefs[i]*M_PI*std::cos(2*(i+1)*M_PI*j*h)/h;
			    		qlist2[j] -= (i+1)*ratio*strength*coefs[i]*M_PI*std::cos(2*(i+1)*M_PI*(j+1)*h)/h;
			 			}
					}
				auto ampo2 = AutoMPO(sites);
				for(auto j : range1(bins))
					{
					ampo2 += plist2[j-1],"S+",j,"S-",j%bins+1;
					ampo2 += -plist2[j-1],"projDn",j,"projUp",j%bins+1;
					ampo2 += qlist2[j-1],"S-",j,"S+",j%bins+1;
					ampo2 += -qlist2[j-1],"projUp",j,"projDn",j%bins+1;
					}
				W2 = toMPO(ampo2);
				for(auto i : range(np+1))
					{
					if(i<=bins/2 && np-i<=bins/2)
						{
						BinaryTree smalltree1 = smalltreelist[i]*std::sqrt((double)choose(bins/2,i)/choose(bins,np));
						BinaryTree smalltree2 = smalltreelist[np-i]*std::sqrt((double)choose(bins/2,np-i));
						// printfln("doubleTree %d %d",np,i);

						// println("before");
						// if(largetree) PrintData(largetree(1).inds());
						largetree = sum(largetree,doubleTree(smalltree1,smalltree2,state),{"Cutoff",co1});
						// println("after");
						// if(largetree) PrintData(largetree(1).inds());

						// largetree.orthogonalize({"Cutoff",co1,"MaxDim",maxdim});
						// println("after2");
						// if(largetree) PrintData(largetree(1).inds());

						// println(inner(largetree,W2,largetree));
						// println(norm(largetree));
						// for(auto n : range1(bins)) printf("%f ",siteval(largetree,n)[1]);
						// println();

						// if(bins==8)
						// 	{
						// 	auto Hfull = smalltree1(0) * smalltree1(1) * smalltree1(2)/* * smalltree1(3) * smalltree1(4) * smalltree1(5) * smalltree1(6)*/;
						// 	auto inds = Hfull.inds();
						// 	auto C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3]/*, inds[4], inds[5], inds[6], inds[7]*/));
						// 	auto psi1fullmat = C * Hfull;
						// 	PrintData(psi1fullmat);
						// 	println(norm(smalltree1));

						// 	Hfull = smalltree2(0) * smalltree2(1) * smalltree2(2)/* * smalltree2(3) * smalltree2(4) * smalltree2(5) * smalltree2(6)*/;
						// 	inds = Hfull.inds();
						// 	C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3]/*, inds[4], inds[5], inds[6], inds[7]*/));
						// 	psi1fullmat = C * Hfull;
						// 	PrintData(psi1fullmat);
						// 	println(norm(smalltree2));

						// 	Hfull = largetree(0) * largetree(1) * largetree(2) * largetree(3) * largetree(4) * largetree(5) * largetree(6)/* * largetree(7) * largetree(8) * largetree(9) * largetree(10) * largetree(11) * largetree(12) * largetree(13) * largetree(14)*/;
						// 	inds = Hfull.inds();
						// 	C = std::get<0>(combiner(inds[0], inds[1], inds[2], inds[3], inds[4], inds[5], inds[6], inds[7]/*, inds[8], inds[9], inds[10], inds[11], inds[12], inds[13], inds[14], inds[15]*/));
						// 	psi1fullmat = C * Hfull;
						// 	PrintData(psi1fullmat);
						// 	println(norm(largetree));
						// 	}
						}
					}
				largetree.normalize();

				// for(auto j : range(largetree.size()))
				// 	{
				// 	println(j);
				// 	PrintData(largetree(j).inds());
				// 	}

				// printfln("before ortho %d",np);
				// println(inner(largetree,W2,largetree));
				// println(norm(largetree));
				// for(auto n : range1(bins)) printf("%f ",siteval(largetree,n)[1]);
				// println();
				// largetree.orthoReset();
				// // largetree.orthogonalize({"Cutoff",co1});
				// largetree.position(largetree.endPoint(),{"Cutoff",co1});
				// largetree.orthoReset();
				// largetree.position(largetree.startPoint(),{"Cutoff",co1});
				// printfln("after ortho %d",np);
				// println(inner(largetree,W2,largetree));
				// println(norm(largetree));
				// for(auto n : range1(bins)) printf("%f ",siteval(largetree,n)[1]);
				// println();

				largetreelist.push_back(largetree);
				}
			smalltreelist.clear();
			}
		else psi0 = smalltreelist.front();
		}

	auto period = 1/freq;
	auto transient_freq = std::max(100.0,freq);
	auto transient_period = 1/transient_freq;
	auto nstages1 = std::max(5,(int)(tdvp_freq1*transient_period/2));
	auto deltat1 = transient_period/(2*nstages1);
	auto nstages2 = (int)(tdvp_freq2*(period-transient_period)/2);
	auto deltat2 = nstages2 ? (period-transient_period)/(2*nstages2) : 0.0;

	auto sweeps1 = Sweeps(nstages1);
	sweeps1.maxdim() = maxdim;
	sweeps1.cutoff() = co2;
	sweeps1.niter() = 100;
	sweeps1.noise() = 0.0;
	sweeps1.alpha() = alpha;
	println();
	println(sweeps1);

	auto sweeps2 = Sweeps(nstages2);
	sweeps2.maxdim() = maxdim;
	sweeps2.cutoff() = co2;
	sweeps2.niter() = 100;
	sweeps2.noise() = 0.0;
	sweeps2.alpha() = alpha;
	println();
	println(sweeps2);

	println("\nStart TDVP");

	int maxiter = freq, miniter = freq/100;
	if(maxiter<10) maxiter = 10;
	if(maxiter>1E4) maxiter = 1E4;
	Real mean = 0.0, var = 0.0;
	int iter = 0;
	auto psim = psi0;
	auto psip = psi0;
	while(iter<maxiter)
		{
		// for(auto j : range(bins-1))
		// 	{
		// 	println(j);
		// 	PrintData(psim(j).inds());
		// 	PrintData(psip(j).inds());
		// 	}
		auto psim0 = psim;
		auto psip0 = psip;
		auto mean0 = mean;
		psim = std::get<1>(tree_tdvp(W1m,psim,deltat1,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		if(nstages2) psim = std::get<1>(tree_tdvp(W1m,psim,deltat2,sweeps2,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		psim = std::get<1>(tree_tdvp(W2m,psim,deltat1,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		if(nstages2) psim = std::get<1>(tree_tdvp(W2m,psim,deltat2,sweeps2,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		psip = std::get<1>(tree_tdvp(W1p,psip,deltat1,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		if(nstages2) psip = std::get<1>(tree_tdvp(W1p,psip,deltat2,sweeps2,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		psip = std::get<1>(tree_tdvp(W2p,psip,deltat1,sweeps1,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		if(nstages2) psip = std::get<1>(tree_tdvp(W2p,psip,deltat2,sweeps2,{"NumCenter",1,"DoNormalize",false,"SubspaceExpansion",se2==1,"Quiet",}));
		if(dens>0)
			{
			psi0 = std::get<1>(tree_tdvp(W1,psi0,deltat1,sweeps1,{"NumCenter",1,"SubspaceExpansion",se2==1,"Quiet",}));
			if(nstages2) psi0 = std::get<1>(tree_tdvp(W1,psi0,deltat2,sweeps2,{"NumCenter",1,"SubspaceExpansion",se2==1,"Quiet",}));
			psi0 = std::get<1>(tree_tdvp(W2,psi0,deltat1,sweeps1,{"NumCenter",1,"SubspaceExpansion",se2==1,"Quiet",}));
			if(nstages2) psi0 = std::get<1>(tree_tdvp(W2,psi0,deltat2,sweeps2,{"NumCenter",1,"SubspaceExpansion",se2==1,"Quiet",}));
			}
		auto left = std::log(inner(psim0,psim))/period, right = std::log(inner(psip0,psip))/period;
		mean = (right-left)/(2*dz);
		var = (right+left)/(dz*dz);
		printfln("\n%d: v- = %f, v+ = %f, j = %f, dj = %f",iter++,left,right,mean,mean-mean0);
		psim.normalize();
		psip.normalize();
		Real thresh = std::pow(10,(int)std::log10(fabs(mean)))/1000;
		if(fabs(mean-mean0)<thresh && iter>miniter) break;
		}
	if(iter==maxiter) println("\nMax iterations reached!");
	printfln("\n{jbar, varj} = {%f, %f}",mean,var);

	if(dens>0)
		{
		auto nstages3 = std::max(5,(int)(tdvp_freq1*period/dens));
		auto deltat3 = period/(2*nstages3);
		auto sweeps3 = Sweeps(nstages3);
		sweeps3.maxdim() = maxdim;
		sweeps3.cutoff() = co2;
		sweeps3.niter() = 100;
		sweeps3.noise() = 0.0;
		sweeps3.alpha() = alpha;
		std::ofstream ofs;
		ofs.open("dens_"+std::to_string(nparticles)+"_"+std::to_string((int)freq)+"_"+std::to_string(bins)+".txt");
		for(auto j : range(dens))
			{
			psi0 = std::get<1>(tree_tdvp(j*2/dens==0?W1:W2,psi0,deltat3,sweeps3,{"NumCenter",1,"Quiet",}));
			for(auto n : range1(bins)) ofs << siteval(psi0,n)[1] << " ";
			ofs << std::endl;
			}
		ofs.close();
		}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	printfln("\nElapsed time: %f s",elapsed.count());

	}