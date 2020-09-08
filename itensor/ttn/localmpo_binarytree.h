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
#ifndef __ITENSOR_LocalMPO_BT
#define __ITENSOR_LocalMPO_BT
#include "itensor/mps/mpo.h"
#include "itensor/ttn/localop_tree.h"
#include "itensor/ttn/binarytree.h"


namespace itensor {

  //
  // The LocalMPO_BT class projects an MPO 
  // into the reduced Hilbert space of
  // some number of sites of an BinaryTree.
  // (The default is 2 sites.)
  //
  //   .----...---                ----...--.
  //   |  |     |      |      |     |      | 
  //   W1-W2-..Wj-1 - Wj - Wj+1 -- Wj+2..-WN
  //   |  |     |      |      |     |      | 
  //   '----...---                ----...--'
  //
  // 
  //  Here the W's are the site tensors
  //  of the MPO "Op" and the method position(j,psi)
  //  has been called using the BinaryTree 'psi' as a basis 
  //  for the projection.
  //
  //  This results in an unprojected region of
  //  num_center sites starting at site j.
  //

  class LocalMPO_BT
  {
  public:

    //
    // Constructors
    //

    LocalMPO_BT();

    //
    //Regular case where H is an MPO for a finite system
    //
    LocalMPO_BT(MPO const& H, 
		Args const& args = Args::global());

    //
    //Use an MPS instead of an MPO. Equivalent to using an MPO
    //of the outer product |Psi><Psi| but much more efficient.
    // 
    LocalMPO_BT(MPS const& Psi, 
		Args const& args = Args::global());

    //
    // Sparse Matrix Methods
    //

    void
    product(const ITensor& phi, ITensor& phip) const;

    Real
    expect(const ITensor& phi) const { return lop_.expect(phi); }

    ITensor
    deltaRho(const ITensor& AA, 
             const ITensor& comb, Direction dir) const
    { return lop_.deltaRho(AA,comb,dir); }

    ITensor
    diag() const { return lop_.diag(); }

    //
    // position(b,psi) uses the BinaryTree psi
    // to adjust the edge tensors such
    // that the MPO tensors at positions
    // b and parent(b) are exposed
    //
    void
    position(int b, BinaryTree const& psi);

    //
    // Accessor Methods
    //

    void
    reset()
    {
      std::fill(Hlim_.begin(), Hlim_.end(), -1);
    }

    const MPO&
    H() const 
    { 
      if(Op_ == 0)
	Error("LocalMPO_BT is null or contains an BinaryTree");
      return *Op_;
    }

    int
    numCenter() const { return nc_; }
    void
    numCenter(int val) 
    { 
      if(val < 0 || val > 2) Error("numCenter must be set 0 or 1 or 2");
      nc_ = val; 
      lop_.numCenter(val);
    }

    size_t
    size() const { return lop_.size(); }

    explicit operator bool() const { return Op_ != 0 || Psi_ != 0; }

    std::vector<int> 
    Lim() const{ return Hlim_;}

    void haveBeenUpdated(int b)
    { 	
      Hlim_.at(b)=-1;
      /*			for (auto & elem : Hlim_)*/
      /*			{if (elem == b) elem=-1;}*/
    }//To be called to say that the tensor number b have been updated

    bool
    doWrite() const { return false; }
    void
    doWrite(bool val,
            Args const& args = Args::global()) 
    { 
      Error("doWrite is not supported by LocalMPO_BT");
    }


  private:

    /////////////////
    //
    // Data Members
    //

    const MPO* Op_;
    std::vector<ITensor> PH_;
    std::vector<int> Hlim_; // That is the limits value of the product
    int nc_;

    LocalOpTree lop_;

    const MPS* Psi_;

    //
    /////////////////

    void
    makeHloc(const BinaryTree& psi, int k);

  };

  inline LocalMPO_BT::
  LocalMPO_BT()
    : Op_(0),
      Hlim_(0),
      nc_(0),
      Psi_(0)
  { }

  inline LocalMPO_BT::
  LocalMPO_BT(const MPO& H, 
	      const Args& args)
    : Op_(&H),
      PH_(2*H.length()-1),
      Hlim_(2*H.length()-1,-2),
      Psi_(0)
  { 
    if(args.defined("NumCenter"))
      numCenter(args.getInt("NumCenter"));
    for(auto i: range1(H.length()))
      {
	PH_[H.length()-2+i] = Op_->A(i);
      }
      nc_ = args.getInt("NumCenter",2);
  }

  inline LocalMPO_BT::
  LocalMPO_BT(const MPS& Psi, 
	      const Args& args)
    : Op_(0),
      PH_(Psi.length()-1),
      Hlim_(Psi.length()-1,-2),
      Psi_(&Psi)
  { 
    if(args.defined("NumCenter"))
      numCenter(args.getInt("NumCenter"));
    for(auto i: range1(Psi.length()))
      {
	PH_[Psi.length()-2+i] = Psi_->A(i)*dag(prime(Psi_->A(i)));
      }
      nc_ = args.getInt("NumCenter",2);
  }


  void inline LocalMPO_BT::
  product(ITensor const& phi, 
	  ITensor& phip) const
  {
    if(Op_ != 0 || Psi_ != 0)
      {
	lop_.product(phi,phip);
      }
    else
      {
        Error("LocalMPO_BT is null");
      }
  }

  inline void LocalMPO_BT::
  position(int b, BinaryTree const& psi)
  {
    if(!(*this)) Error("LocalMPO_BT is null");
		
    makeHloc(psi,b);

    if(nc_ == 2 && b == 0 )
      {
        Error("LocalMPO_BT position cannot position at 0 with 2 center sites");
      }

#ifdef DEBUG
    if(nc_ != 2 && nc_ != 1 )
      {
        Error("LocalMPO_BT only supports 1 and 2 center sites currently");
      }
#endif

    if(Op_ != 0) //normal MPO case //TODO Update LocalOpTree to not be dependent of the structure of the tree
      {
        if(nc_ == 2)
	  {
	    if( psi.parent(b) == 0)//The if the top node is inclued this is a particular case
	      {
		lop_.update(PH_.at(2*b+2), PH_.at(2*b+1), PH_.at(psi.sibling(b)));
	      }
	    else
	      {
		lop_.update(PH_.at(2*b+2), PH_.at(psi.sibling(b)),PH_.at(2*b+1),PH_.at(psi.parent(psi.parent(b)))); 
	      }
	  }
        else if(nc_ == 1)
	  {
	    if (b == 0)  // If we updated the top node
	      {
		lop_.update(PH_.at(2*b+2), PH_.at(2*b+1));
	      }
	    else
	      {
            	lop_.update(PH_.at(2*b+2), PH_.at(2*b+1), PH_.at(psi.parent(b)));
	      }
	  }
      }
  }


  inline void LocalMPO_BT::
  makeHloc(BinaryTree const& psi, int k)
  {
    if(!PH_.empty())
      {
	//Find the max distance from the position to contract
	auto dist_to_zero=psi.depth(k);
	auto max_dist=psi.height()+dist_to_zero;
	//By decreasing distance, we check if the tensor are contracted, if not we contracted them
	for(int d =max_dist; d > 0; d--) //Max distance is zero as we do not want to contract into the node k
	  {
	    //println(d);
	    auto node_d = psi.node_list(k,d); // Get the list of node to check, each element is the node and the node towards it is supposed to point out
	    for(unsigned int i=0; i < node_d.size(); i++) 
	      {
		auto node=node_d.at(i)[0];
		auto direction=node_d.at(i)[1];

		if ((d==1 && node == psi.parent(k) )&& nc_ ==2 ) continue; // If numCenter is two we avoid one unneeded contraction
				
		if (Hlim_.at(node) != direction) // That check if the contracted point out in the correct direction. If not then contract, if yes do nothing, this point save computer time, as it does not recompute product already computed
		  {
		    auto to_contract=psi.othersLinks(node, direction);
						
		    //println("Contraction: ",node," ",direction);

		    PH_.at(node)=psi(node);
					
		    for (auto & conc : to_contract)
		      {
			PH_.at(node)*=PH_.at(conc);
		      }

		    PH_.at(node)*=dag(prime(psi(node)));
		    //println(inds(PH_.at(node)));

		    //We update the current status of contraction 
		    Hlim_.at(node) = direction;// This node have now open indices towards direction
		    Hlim_.at(direction) = -1; // This is the next one to be updated

		  }
	      }
	  }
      }
  }


} //namespace itensor


#endif
