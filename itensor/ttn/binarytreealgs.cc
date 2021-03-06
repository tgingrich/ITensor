//
// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "itensor/ttn/binarytree.h"
// #include "itensor/mps/mpo.h"
#include "itensor/util/print_macro.h"
#include "itensor/ttn/localmpo_binarytree.h"
// #include "itensor/tensor/slicemat.h"

namespace itensor {

using std::istream;
using std::ostream;
using std::cout;
using std::endl;
using std::vector;
// using std::find;
// using std::pair;
// using std::make_pair;
// using std::string;
// using std::move;

long subspace_expansion(BinaryTree & psi,LocalMPO_BT & PH,int b1,int b2, Real alpha, bool ortho)
{
    //Build Pi
    ITensor Pi = alpha*dag(prime(psi(b1)));
    auto neighbors=psi.othersLinks(b1,b2);
    for(unsigned int i=0; i < neighbors.size(); ++i)
    {
        // PrintData(Pi);
        // PrintData(PH(neighbors.at(i)));
        Pi*=PH(neighbors.at(i));
    }
    // PrintData(Pi);
    auto ind_b1=commonIndex(psi(b1),psi(b2));
    auto inds_b2 =uniqueInds(psi(b2), psi(b1));
    auto original_link_tags = tags(ind_b1);
    auto tocombine_inds=uniqueInds(Pi,psi(b1));
    // Use of combiner to mix indexes
    auto [Comb,extra_ind] = ortho ? combiner(tocombine_inds) : combiner(dag(tocombine_inds));
    auto PiC = ortho ? dag(Pi*Comb) : dag(Pi)*Comb;
    //Expand psi(b1)

    // PrintData(tocombine_inds);
    // PrintData(Comb);
    // PrintData(PiC);
    auto [ExtentedTensor,ind] = directSum(psi(b1),PiC,ind_b1,dag(extra_ind));
    auto new_ind=ind;
    new_ind.setTags(original_link_tags);
    ExtentedTensor.replaceInds(IndexSet(ind),IndexSet(new_ind));
    psi.ref(b1)=ExtentedTensor;
    
    //Build zero block with correct indexes that are the extra one, and the two other of psi(b2)
    ITensor zero;
    if(hasQNs(psi(b2)))
    {
        // We need to allocate the zero tensor at the same divergence
        zero = ortho ? ITensor(div(psi(b2)),unionInds(inds_b2,extra_ind)) : ITensor(div(psi(b2)),unionInds(inds_b2,dag(extra_ind)));
    }
    else
    {
        zero = ITensor(unionInds(inds_b2,extra_ind));
    }
    zero.fill(0.);

    // PrintData(psi(b2));
    // PrintData(zero);
    auto [ExtentedTensorBis,indBis] = directSum(psi(b2),zero,dag(ind_b1),extra_ind);
    ExtentedTensorBis.replaceInds(IndexSet(indBis),IndexSet(dag(new_ind)));
    psi.ref(b2)=ExtentedTensorBis;
    return new_ind.dim();
}

bool
checkQNs(BinaryTree const& psi)
    {
    const int N = size(psi);

    QN Zero;

    int center = findCenter(psi);
    if(center == -1)
        {
        cout << "Did not find an ortho. center\n";
        return false;
        }

    //Check that all IQTensors have zero div
    //except possibly the ortho. center
    for(int i = 0; i < N; ++i)
        {
        if(psi.orthoPos(i) == -1) continue;
        if(!psi(i))
            {
            println("A(",i,") null, QNs not well defined");
            return false;
            }
        if(div(psi(i)) != Zero)
            {
            cout << "At i = " << i << "\n";
            Print(psi(i));
            cout << "ITensor other than the ortho center had non-zero divergence\n";
            return false;
            }
        }

    //Check all arrows
    for(int i = 1; i < N; ++i)
        {
            if (psi.orthoPos(i) == -1) continue; //We are at the orthogonality center
            if(psi.orthoPos(i) == psi.parent(i))
            {
                if(dir(linkIndex(psi,i)) != In)
                {
                    println("checkQNs: At site ",i," Link not pointing In");
                    return false;
                }
            }
            else if(psi.orthoPos(psi.parent(i)) == i)
            {
                if(dir(linkIndex(psi,i)) != Out)
                {
                    println("checkQNs: At site ",i," Link not pointing Out");
                    return false;
                }
            }
            else
            {
                println("checkQNs: At site ",i,"unable to determine direction for orthogonality direction ",psi.orthoPos(i), "and parent direction ",psi.orthoPos(psi.parent(i)));
                return false;
            }
        }
    //Done checking arrows
    return true;
    }

QN
totalQN(BinaryTree const& psi)
    {
    auto tq = QN();
    const int N = psi.size();
    for(int j = 0; j < N; ++j)
        {
        tq += div(psi(j));
        }
    return tq;
    }

} //namespace itensor
