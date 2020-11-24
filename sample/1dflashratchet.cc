#include "itensor/all.h"
#include "itensor/ttn/binarytree.h"
#include "itensor/ttn/tree_dmrg.h"
#include "itensor/ttn/tree_tdvp.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(int argc, char** argv)
	{
	auto freq = stod(argv[3]);
	int nparticles = atoi(argv[2]);
	ifstream ifs;
	ifs.open(argv[1]);
	std::string type, val;
	Real len, strength, ratio, di;
	int bins, nstages;
	int idx = 0;
	while(ifs >> type >> val)
		{
		switch(idx++)
			{
			case 0:
				len = atod(val);
			case 1:
				bins = atoi(val);
			case 2:
				strength = atod(val);
			case 3:
				ratio = atod(val);
			case 4:
				di = atod(val);
			case 5:
				nstages = atoi(val);
			}
		}
	auto h = len / bins;
	ifs.close();
	ifs.open("coefs.in");
	std::vector<Real> coefs;
	while(ifs >> val)
		{
		coefs.push_back(stod(val));
		}
	ifs.close();
	}