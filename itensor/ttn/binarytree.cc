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
#include "itensor/mps/localop.h"
#include <stack>

namespace itensor {

  using std::istream;
  using std::ostream;
  using std::cout;
  using std::endl;
  using std::vector;
  using std::stack;
  using std::find;
  using std::pair;
  using std::make_pair;
  using std::string;

  void
  binary_new_tensors(std::vector<ITensor>& A,
		     SiteSet const& sites,
		     int m = 1);

  void
  binary_new_tensors(std::vector<ITensor>& A,
		     IndexSet const& sites,
		     int m = 1);

  //
  // class BinaryTree
  //

  //
  // Constructors
  //

  BinaryTree::
  BinaryTree():N_(0),height_(0),N_sites_(0),orth_pos_(0), order_(0), reverse_order_(0), site_dim_(0)
  { }

  BinaryTree::
  BinaryTree(int mlength):
    N_(mlength-1),  height_(intlog2(mlength)-1), N_sites_(mlength),
    A_(mlength-1), //2^(h+1)-1
    orth_pos_(mlength-1,-1), order_(mlength-1), reverse_order_(mlength-1), site_dim_(2)
  {
    if(!islog2(mlength)) Error("The number of sites is not a power of 2");
    for(int i =0; i < N_; ++i)
      {
	order_[i] = i-1;
	reverse_order_[i] = i+1;
      }
    reverse_order_.at(N_-1) = -1*N_;
  }

  BinaryTree::BinaryTree(SiteSet const& sites,int m)
    : N_(sites.length()-1),  height_(intlog2(sites.length())-1), N_sites_(sites.length()),
      A_(sites.length()-1), //2^(h+1)-1
      orth_pos_(sites.length()-1,-1),order_(sites.length()-1), reverse_order_(sites.length()-1), site_dim_(sites.inds()[0].dim())
  {
    if(!islog2(sites.length())) Error("The number of sites is not a power of 2");
    binary_new_tensors(A_,sites,m);
    for(int i =0; i < N_; ++i)
      {
	order_[i] = i-1;
	reverse_order_[i] = i+1;
      }
    reverse_order_.at(N_-1) = -1*N_;
  }

  BinaryTree::BinaryTree(IndexSet const& sites,int m)
    : N_(sites.length()-1),  height_(intlog2(sites.length())-1), N_sites_(sites.length()),
      A_(2*sites.length()-1),  //2*2^(h)-1
      orth_pos_(sites.length()-1,-1), order_(sites.length()-1), reverse_order_(sites.length()-1), site_dim_(sites[0].dim())
  {
    if(!islog2(sites.length())) Error("The number of sites is not a power of 2");
    binary_new_tensors(A_,sites,m);
    for(int i =0; i < N_; ++i)
      {
	order_[i] = i-1;
	reverse_order_[i] = i+1;
      }
    reverse_order_.at(N_-1) = -1*N_;
  }

  BinaryTree::
  BinaryTree(InitState const& initState)
    : N_(initState.sites().length()-1),  height_(intlog2(initState.sites().length())-1), N_sites_(initState.sites().length()),
      A_(2*initState.sites().length()-1), //2*2^(h)-1
      orth_pos_(initState.sites().length()-1,-1),
      order_(initState.sites().length()-1), reverse_order_(initState.sites().length()-1), site_dim_(initState.sites().inds()[0].dim())
  {
    if(!islog2(initState.sites().length())) Error("The number of sites is not a power of 2");
    init_tensors(initState);
    for(int i =0; i < N_; ++i)
      {
	order_[i] = i-1;
	reverse_order_[i] = i+1;
      }
    reverse_order_.at(N_-1) = -1*N_;
  }


  BinaryTree::BinaryTree(BinaryTree const& other)
  {
    N_ = other.N_;
    height_ = other.height_;
    N_sites_ = other.N_sites_;
    A_ = other.A_;
    orth_pos_= other.orth_pos_;
    order_=other.order_;
    reverse_order_=other.reverse_order_;
    site_dim_ = other.site_dim_;
  }

  BinaryTree& BinaryTree::
  operator=(BinaryTree const& other)
  {
    N_ = other.N_;
    height_ = other.height_;
    N_sites_ = other.N_sites_;
    A_ = other.A_;
    orth_pos_= other.orth_pos_;
    order_=other.order_;
    reverse_order_=other.reverse_order_;
    site_dim_ = other.site_dim_;
    return *this;
  }


  // Helpers for the constructors


  BinaryTree::~BinaryTree(){}

  void binary_new_tensors(std::vector<ITensor>& A,
			  SiteSet const& sites,
			  int m)
  {
    auto N = length(sites)-1;// Number of nodes that is 2^(h+1)-1
    //auto N_sites=length(sites); // Number of sites
    auto height=intlog2(length(sites))-1;//Compute the height from the number of sites
    auto a = std::vector<Index>(N); //Number of link to be created is N-1 but we skip the first one

    if(hasQNs(sites))
      {
        if(m==1) for(auto i : range(N)) a[i] = Index(QN(),m,format("Link,l=%d",i));
        else Error("Cannot create QN conserving BinaryTree with bond dimension greater than 1 from a SiteSet");
      }
    else
      {
        for(auto i : range(N)) a[i] = Index(m,format("Link,l=%d",i));
      }

    //First tensor that is only a rank-2 tensor
    A[0] = ITensor(a[1],a[2]);

    for(int i = 1; i < N/2; ++i) //Ony the middle ones
      {
        A[i] = ITensor(dag(a[i]),a[2*i+1],a[2*i+2]);
      }
    //Last line connection to the sites
    for(int i=0; i < pow2(height); ++i) //That correspond to the last line
      {
	A[pow2(height)-1+i] = ITensor(a[pow2(height)-1+i],sites(2*i+1),sites(2*i+2)); //Sites are 1 to N
      }
  }

  void binary_new_tensors(std::vector<ITensor>& A,
			  IndexSet const& sites,
			  int m)
  {
    auto N = length(sites)-1;// Number of nodes that is 2^(h+1)-1
    //auto N_sites=length(sites); // Number of sites
    auto height=intlog2(length(sites))-1;//Compute the height from the number of sites
    auto a = std::vector<Index>(N); //Number of link to be created is N-1 but we skip the first one
    if(hasQNs(sites))
      {
	if(m==1) for(auto i : range(N)) a[i] = Index(QN(),m,format("Link,l=%d",i));
	printfln("Cannot create QN conserving BinaryTree with bond dimension greater than 1 from an IndexSet");
      }
    for(auto i : range1(N)) a[i] = Index(m,format("Link,l=%d",i));
    //First tensor that is only a rank-2 tensor
    A[0] = ITensor(a[1],a[2]);

    for(int i = 1; i < N/2; ++i) //Ony the middle ones
      {
        A[i] = ITensor(dag(a[i]),a[2*i+1],a[2*i+2]);
      }
    //Last line connection to the sites
    for(int i=0; i < pow2(height); ++i) //That correspond to the last line
      {
	A[pow2(height)-1+i] = ITensor(a[pow2(height)-1+i],sites(2*i+1),sites(2*i+2)); //Sites are 1 to N
      }
  }

  void BinaryTree::
  init_tensors(InitState const& initState)
  {
    auto N = initState.sites().length()-1;// Number of nodes that is 2^(h+1)-1
    auto N_sites=initState.sites().length(); // Number of sites
    auto height=intlog2(initState.sites().length())-1;//Compute the height from the number of sites
    auto a = std::vector<Index>(N); //Number of link to be created is N-1 but we skip the first one
    if(hasQNs(initState))
      {
        //for(auto i : range(N)) a[i] = Index(QN(),1,format("Link,l=%d",i));
	auto qa = std::vector<QN>(N); //qn[i] = qn on i^th bond
	//Last line connection to the sites
	for(int i=0; i < N_sites/2; ++i) //That correspond the the N_sites /2 of the last line
	  {
	    qa[pow2(height)+i-1] =Out*(- qn(initState(2*i+1))- qn(initState(2*i+2)));
	  }
	for(int i = N/2-1; i >=0; i--) //Ony the middle ones
	  {
	    qa[i] =Out*(- qa[2*i+1]*In- qa[2*i+2]*In);
	  }
       	for(auto i : range(N))	a[i] = Index(qa[i],1,format("Link,l=%d",i));
      }
    else
      {
        for(auto i : range(N)) a[i] = Index(1,format("Link,l=%d",i));
      }

    A_[0] = setElt(a[1](1),a[2](1));
    for(int i = 1; i < N/2; ++i) //Ony the middle ones
      {
        A_[i] = setElt(itensor::dag(a[i])(1),a[2*i+1](1),a[2*i+2](1));
      }
    //Last line connection to the sites
    for(int i=0; i < N_sites/2; ++i) //That correspond the the N_sites /2 of the last line
      {
    	A_[pow2(height)+i-1] = setElt(itensor::dag(a[pow2(height)+i-1])(1),initState(2*i+1),initState(2*i+2));
      }
  }


  BinaryTree& BinaryTree::
  randomize(Args const& args)
  {
    //if(maxLinkDim(*this)>1) Error(".randomize() not currently supported on BinaryTree with bond dimension greater than 1.");
    for(auto i : range(N_))
      {
	ref(i).randomize(args);
      }
    return *this;
  }

  //
  // BinaryTree Helper function
  //

  std::vector<int> BinaryTree::children(int i,int distance) const{ // Return all the children of the node that are at a certain distance, there is 2^distance of them, we get the first one by descending along the tree and the other one by horizontal displacement
    if (distance <= 0){
      std::vector<int> child(1,i);
      return child;
    }
    if( depth(i)+distance > height_){ // There is no children at this distance
      std::vector<int> child(0);
      return child;
    }
    auto offset=i;
    for (int d=1; d <= distance ; d++)
      {
	offset=2*offset+1;
      }
    std::vector<int> child;
    for(int i=0; i < pow2(distance); ++i)
      {
	child.push_back(offset+i);
      }

    return child;
  }

  std::vector<int[2]> BinaryTree::node_list(int i, int distance) const{
    // Return all the nodes that are at a certain distance from a given node as well as the predecessor in the graph
    // This is all the children at a given distance from every parent of the node for the other subgraph
    // So we iterate to the root and get the children at the reduced distance for each node in the road
    auto de=this->depth(i);
    std::vector<int> node_list=this->children(i,distance);
    std::vector<int> node_pred(node_list.size(),-1);

    auto new_node=i;
    for( int d =1; d <=de; d++) // We start from the node and get the parent then the other children of the parents
      {
	if( distance < d or new_node== -1) break;// No more node to add
	std::vector<int> nodes_to_add(0);
	std::vector<int> pred_to_add(0);

	if (distance > d)
	  {	nodes_to_add = this->children(this->sibling(new_node),distance-d-1);
	    pred_to_add.resize(nodes_to_add.size());
	    fill(pred_to_add.begin(), pred_to_add.end(), -1);
	  }
	else if (distance == d) {
	  nodes_to_add.push_back(this->parent(new_node));
	  pred_to_add.push_back(new_node);
	}
	new_node=this->parent(new_node);
	node_list.insert(node_list.end(), nodes_to_add.begin(), nodes_to_add.end());
	node_pred.insert(node_pred.end(), pred_to_add.begin(), pred_to_add.end());
      }

    std::vector<int[2]> nodes_out(node_list.size());
    for(unsigned int i=0; i < node_list.size(); ++i)
      {
	nodes_out.at(i)[0] =node_list.at(i);
	if(node_pred.at(i) == -1)
	  {
	    nodes_out.at(i)[1] = this->parent(node_list.at(i));
	  }
	else
	  {
	    nodes_out.at(i)[1] = node_pred.at(i);
	  }
      }

    return nodes_out;
  }

  //
  // BinaryTree Accessor Methods
  //

  ITensor const& BinaryTree::
  operator()(int i) const
  {
    if(i < 0) i = N_+i+1;
		if(i >= N_) Error("Attempt to access tensor out of the tree");
    return A_.at(i);
  }

  ITensor& BinaryTree::
  ref(int i)
  {
    if(i < 0) i = N_+i+1;
    //orth_pos_[i]=-1; // The site is not longer orthogonalized but that is up to the dev to check this
    return A_.at(i);
  }

  Real BinaryTree::
  normalize()
  {
    auto nrm = norm(*this);
    if(std::fabs(nrm) < 1E-20) Error("Zero norm");
    *this /= nrm;
    return nrm;
  }


  //
  // Orthogonalization
  //

  BinaryTree& BinaryTree::
  position(int k, Args args)
  {
    if(not *this) Error("position: BinaryTree is default constructed");

    //Find the max distance from the position to orthogonalize
    auto dist_to_zero=depth(k);
    auto max_dist=height_+dist_to_zero;
    //By decreasing distance, we check if the tensor are orthogonal, if not we orthogonalize them
    for(int d =max_dist; d > 0; d--) {
      auto node_d = this->node_list(k,d); // Get the list of node to check, each element is the node and the node towards it is supposed to point out
      for(unsigned int i=0; i < node_d.size(); ++i) {
        if (orth_pos_.at(node_d.at(i)[0]) != node_d.at(i)[1] ) {
          if(args.getBool("DoSVDBond",false)) {
            auto WF = operator()(node_d.at(i)[0]) * operator()(node_d.at(i)[1]);
            svdBond(node_d.at(i)[0],WF,node_d.at(i)[1],args);
          } else {
            orthPair(ref(node_d.at(i)[0]),ref(node_d.at(i)[1]),args);
          }
					orth_pos_.at(node_d.at(i)[0]) = node_d.at(i)[1];// We update the orthogonalisation memory
					orth_pos_.at(node_d.at(i)[1]) = -1; // The next one is not any more orthogonal
        }
      }
    }
    return *this;
  }

  BinaryTree& BinaryTree::
  orthogonalize(Args args) // Since position check the orthognality along the path
  {
    return position(1,args);
  }


  Spectrum BinaryTree::
  svdBond(int b1, ITensor const& AA, int b2,  Args args)
  {
    return svdBond(b1,AA,b2,LocalOp(),args);
  }

  int BinaryTree::
  startPoint(Args const& args) const {
    auto chosenOrder=args.getString("Order","Default");// Default (breath first) is the default value
		const int numCenter = args.getInt("NumCenter",1);
		if (chosenOrder == "Default" ) {
			if(numCenter != 1)
      	return 1;
			return 0;
    } else {
      return N_ / 2;
    }
  }

  //
  // Travel method
  //

  void BinaryTree::
  setOrder(Args const& args) // Set the chosen direction of travel
  // We have the choices: Default, PostOrder, PreOrder, InOrder
  {
    auto chosenOrder=args.getString("Order","Default");// Default (breath first) is the default value
    if (chosenOrder == "Default" ) {
      for(int i =0; i < N_; ++i) {
        order_[i] = i+1;
        reverse_order_[i] = i-1;
      }
      order_[N_-1]=-1*N_;

    } else if (chosenOrder == "PostOrder") {
      // std::stack<int> stack, out_stack;
      // stack.push(0);
      // int origin_node=0;

      // while (!stack.empty()) {
      //   int node = stack.top();
      //   stack.pop();

      //   out_stack.push(node);

      //   auto rightchild = this->rightchild(node);
      //   if (rightchild > 0) {
      //     stack.push(rightchild);
	     //  }

      //   auto leftchild = this->leftchild(node);
      //   if (leftchild > 0) {
      //     stack.push(leftchild);
	     //  }
      // }

      // while (!out_stack.empty()) {
      //   int node = out_stack.top();
      //   out_stack.pop();

      //   order_[origin_node]=node;
      //   origin_node=node;
      // }
      // order_[0]=-1;
      // for(int i =1; i < N_-1; ++i) {
      //   reverse_order_[reverse_order_[i]] = i;
      // }
      // reverse_order_[N_-1]=-1*N_;
      std::stack<int> stack;
      std::vector<int> data;
      int root = 0;
      do {
        while (root >= 0) {
          auto rightchild = this->rightchild(root);
          if (rightchild >= 0) {
            stack.push(rightchild);
          }
          stack.push(root);
          root = this->leftchild(root);
        }
        root = stack.top();
        stack.pop();
        if (!stack.empty() && this->rightchild(root) == stack.top()) {
          int temp = root;
          root = stack.top();
          stack.pop();
          stack.push(temp);
        } else {
          data.push_back(root);
          root = -1;
        }
      } while (!stack.empty());
      order_[0] = -1;
      for(int i = 0; i < N_ - 1; ++i) {
        order_[data[i]] = data[i + 1];
        reverse_order_[data[i + 1]] = data[i];
      }
      reverse_order_[N_ - 1] = -N_;

    } else if (chosenOrder == "PreOrder") {
      std::stack<int> stack;
      stack.push(0);
      auto origin_node = 0;
      while (!stack.empty()) {
        int node = stack.top();
        stack.pop();
        order_[origin_node] = node;
        origin_node = node;
        auto rightchild = this->rightchild(node);
        if (rightchild >= 0) {
          stack.push(rightchild);
	      }

        auto leftchild = this->leftchild(node);
        if (leftchild >= 0) {
          stack.push(leftchild);
	      }
      }
      order_[N_ - 1] = -N_;
      for(int i = 0; i < N_ - 1; ++i) {
        reverse_order_[order_[i]] = i;
      }
      reverse_order_[0] = -1;

    } else if (chosenOrder == "InOrder") {
      std::stack<int> stack;
      // int curr = 0;
      // int origin_node=0;
      // while (!stack.empty() || curr != -1) {
      //   if (curr != -1) {
      //     stack.push(curr);
      //     curr = this->rightchild(curr);
      //   } else {
      //     curr = stack.top();
      //     stack.pop();

      //     order_[origin_node]=curr;
      //     origin_node=curr;
      //     curr = this->leftchild(curr);
      //   }
      // }
      // order_[N_/2] = -1*(N_/2+1);
      // for(int i =0; i < N_-1; ++i) {
      //   reverse_order_[reverse_order_[i]] = i;
      // }
      // reverse_order_[N_-1]=-1*N_;
      std::vector<int> data;
      int root = 0;
      do {
        while (root >= 0) {
          stack.push(root);
          root = this->leftchild(root);
        }
        if (root < 0 && !stack.empty()) {
          root = stack.top();
          stack.pop();
          data.push_back(root);
          root = this->rightchild(root);
        }
      } while(root >= 0 || !stack.empty());
      order_[N_ - 1] = -N_;
      for(int i = 0; i < N_ - 1; ++i) {
        order_[data[i]] = data[i + 1];
        reverse_order_[data[i + 1]] = data[i];
      }
      reverse_order_[N_ / 2] = -N_ / 2 - 1;

    } else {
      Error("setOrder: the required order is not part of Default,PostOrder,PreOrder,InOrder");
    }
  }

  void BinaryTree::setOrder(std::vector<int> new_order) // Among the condition an order should contain at least a position with -1 to signal the end of the travel
  {
    bool find_negative=false;
    for(unsigned int i=1; i < new_order.size();i ++)
      {
	if (new_order[i] < 0)
	  { find_negative= true;
	    break;
	  }
      }
    if(!find_negative) { // We should have a least one postion with a negative value
      Error("setOrder: The new_order should contains a negative value to stop the iteration");
    }
    order_=new_order;
    //Update reverse order
    for(unsigned int i=1; i < new_order.size();i ++)
      {
	reverse_order_[new_order.at(i)] = i;
      }
    reverse_order_[N_-1] = -1*N_;
  }

  void BinaryTree::sweepnext(int &b, int &ha,Args const& args)
  {
    const int numCenter = args.getInt("NumCenter",2);
    const bool reverse = args.getBool("Reverse",false);// By default we do not sweep back

	//println("Sweep ",b," ",ha);
    b= (ha % 2== 1 ? order_.at(b) : reverse_order_.at(b));
    // odd ha means foward order and even one means reverse order
    if(b < 0 || ((ha % 2== 1? order_.at(b) : reverse_order_.at(b)) < 0 && numCenter==2)) // At the end, either we stop or we return in the reverse direction
      {
	b= -1*b-1;//The negative indicate where to start for the reverse (-1-> 0 , -2 -> 1 and so forth
	if (reverse)
	  {
	    ++ha;
	    if (numCenter == 2) b= (ha % 2== 1 ? order_.at(b) : reverse_order_.at(b)); // We avance once more for the double center
	  }
	else{ha+=2;} // We skip the reverse part


      }
	//println("Sweep next ",b," ",ha);
  }

  BinaryTree
  randomBinaryTree(SiteSet const& sites, int m, Args const& args)
  {
    auto psi = BinaryTree(sites);

    auto N = sites.length()-1;// Number of nodes that is 2^(h+1)-1
    auto N_sites=sites.length(); // Number of sites
    auto height=intlog2(sites.length())-1;//Compute the height from the number of sites
    auto a = std::vector<Index>(N); //Number of link to be created is N-1 but we skip the first one
    // auto b = std::vector<Index>(N_sites);
    for(auto i : range(N)) {
      a[i] = Index(std::min((int)std::pow(sites.inds()(1).dim(), pow2(psi.height() - psi.depth(i) + 1)), (int)m),format("Link,l=%d",i));
    }
    // for(auto i : range(N_sites)) b[i] = Index(sites.inds()(1).dim(),format("Site,n=%d",i));

    psi.ref(0) = setElt(a[1](1),a[2](1));
      for(int i = 1; i < N/2; ++i) //Ony the middle ones
    {
      psi.ref(i) = setElt(itensor::dag(a[i])(1),a[2*i+1](1),a[2*i+2](1));
    }
    //Last line connection to the sites
    for(int i=0; i < N_sites/2; ++i) //That correspond the the N_sites /2 of the last line
    {
      psi.ref(pow2(height)+i-1) = setElt(itensor::dag(a[pow2(height)+i-1])(1),sites.inds()[2*i](1),sites.inds()[2*i+1](1));
    }

    psi.randomize(args);
    psi /= std::sqrt(inner(psi, psi));
    return psi;
  }

  BinaryTree
  randomBinaryTree(InitState const& initstate, int m, Args const& args)
  {
    if(m>1) Error("randomBinaryTree(InitState,m>1) not currently supported.");
    auto psi = BinaryTree(initstate);
    psi.randomize(args);
    return psi;
  }

  BinaryTree
  randomBinaryTree(InitState const& initstate, Args const& args)
  {
    return randomBinaryTree(initstate,1,args);
  }

  bool
  isOrtho(BinaryTree const& psi)
  {
    int number_center=0;
    std::vector<int> ortho= psi.orthoVect();
    for(unsigned int i=0; i< ortho.size();++i)
      {
	if(ortho.at(i) == -1) number_center++;
      }
    if (number_center == 1) return true;
    return false;
  }

  int orthoCenter(BinaryTree const& psi)
  {
    if(!isOrtho(psi)) Error("orthogonality center not well defined.");
    auto ortho= psi.orthoVect();
    for(unsigned int i=0; i< ortho.size();++i)
      {
	if(ortho.at(i) == -1) return i;
      }
    return -1;
  }

	int findCenter(BinaryTree const& psi)
	{
		if(!isOrtho(psi)) return -1;
    auto ortho= psi.orthoVect();
    for(unsigned int i=0; i< ortho.size();++i)
      {
	if(ortho.at(i) == -1) return i;
      }
    return -1;
	}


  //
  // Operator
  //


  BinaryTree&
  operator*=(BinaryTree & x, Real a) { x.ref(0) *= a; return x; }

  BinaryTree&
  operator/=(BinaryTree & x, Real a) { x.ref(0) /= a; return x; }

  BinaryTree
  operator*(BinaryTree x, Real r) { x *= r; return x; }

  BinaryTree
  operator*(Real r, BinaryTree x) { x *= r; return x; }

  BinaryTree&
  operator*=(BinaryTree & x, Cplx z) { x.ref(0) *= z; return x; }

  BinaryTree&
  operator/=(BinaryTree & x, Cplx z) { x.ref(0) /= z; return x; }

  BinaryTree
  operator*(BinaryTree x, Cplx z) { x *= z; return x; }

  BinaryTree
  operator*(Cplx z, BinaryTree x) { x *= z; return x; }

  BinaryTree
  dag(BinaryTree A)
  {
    A.dag();
    return A;
  }

  int
  length(BinaryTree const& W)
  {
    return W.length();
  }

  int
  size(BinaryTree const& W)
  {
    return W.size();
  }

  bool
  isComplex(BinaryTree const& psi)
  {
    for(auto j : range(psi.size()))
      {
        if(itensor::isComplex(psi(j))) return true;
      }
    return false;
  }


  Real
  norm(BinaryTree const& psi)
  {
    if(not isOrtho(psi)) Error("\
BinaryTree must have well-defined ortho center to compute norm; \
call .position(j) or .orthogonalize() to set ortho center");
    return itensor::norm(psi(findCenter(psi)));
  }

  template<typename TreeType>
  Real
  averageLinkDim(TreeType const& x)
  {
    Real avgdim = 0.;
    auto N = size(x);
    for( auto b : range1(N) )
      {
        avgdim += dim(linkIndex(x,b));
      }
    avgdim /= N;
    return avgdim;
  }
  template Real averageLinkDim<BinaryTree>(BinaryTree const& x);

  template<typename TreeType>
  int
  maxLinkDim(TreeType const& x)
  {
    int maxdim = 0;
    for( auto b : range1(size(x)) )
      {
        int mb = dim(linkIndex(x,b));
        maxdim = std::max(mb,maxdim);
      }
    return maxdim;
  }
  template int maxLinkDim<BinaryTree>(BinaryTree const& x);

  BinaryTree
  setTags(BinaryTree A, TagSet const& ts, IndexSet const& is)
  {
    A.setTags(ts,is);
    return A;
  }

  BinaryTree
  noTags(BinaryTree A, IndexSet const& is)
  {
    A.noTags(is);
    return A;
  }

  BinaryTree
  addTags(BinaryTree A, TagSet const& ts, IndexSet const& is)
  {
    A.addTags(ts,is);
    return A;
  }

  BinaryTree
  removeTags(BinaryTree A, TagSet const& ts, IndexSet const& is)
  {
    A.removeTags(ts,is);
    return A;
  }

  BinaryTree
  replaceTags(BinaryTree A, TagSet const& ts1, TagSet const& ts2, IndexSet const& is)
  {
    A.replaceTags(ts1,ts2,is);
    return A;
  }

  BinaryTree
  swapTags(BinaryTree A, TagSet const& ts1, TagSet const& ts2, IndexSet const& is)
  {
    A.swapTags(ts1,ts2,is);
    return A;
  }

  BinaryTree
  prime(BinaryTree A, int plev, IndexSet const& is)
  {
    A.prime(plev,is);
    return A;
  }

  BinaryTree
  prime(BinaryTree A, IndexSet const& is)
  {
    A.prime(is);
    return A;
  }

  BinaryTree
  setPrime(BinaryTree A, int plev, IndexSet const& is)
  {
    A.setPrime(plev,is);
    return A;
  }

  BinaryTree
  mapPrime(BinaryTree A, int plevold, int plevnew, IndexSet const& is)
  {
    A.mapPrime(plevold,plevnew,is);
    return A;
  }

  BinaryTree
  swapPrime(BinaryTree A, int plevold, int plevnew, IndexSet const& is)
  {
    A.swapPrime(plevold,plevnew,is);
    return A;
  }

  BinaryTree
  noPrime(BinaryTree A, IndexSet const& is)
  {
    A.noPrime(is);
    return A;
  }



  bool
  hasSiteInds(BinaryTree const& x, IndexSet const& sites)
  {
    auto N = length(x);
    if( N!=length(sites) ) Error("In hasSiteInds(BinaryTree,IndexSet), lengths of BinaryTree and IndexSet of site indices don't match");
    for( auto n : range1(N) )
      {
	if( !hasIndex(x(x.bottomLayer(n)),sites(n)) ) return false;
      }
    return true;
  }

  Index
  siteIndex(BinaryTree const& psi, int site) // Should return only the link connected to the site
  {
    return uniqueIndex(psi(psi.bottomLayer(site)),psi(psi.parent(psi.bottomLayer(site))),"n="+str(site));
  }

  template <typename TreeType>
  Index
  linkIndex(TreeType const& psi, int i)
  {
    if(i <=0) return Index();
    return commonIndex(psi(i),psi(psi.parent(i)));
  }
  template Index linkIndex<BinaryTree>(BinaryTree const& W, int i);

  template <typename TreeType>
  IndexSet// Return all links index of the node
  linkInds(TreeType const& x, int i)
  {
    auto parent=x.parent(i);
    auto childs=x.childrens(i);
    std::vector<Index> listInds(0);
    if( parent >=0) listInds.push_back(linkIndex(x,i));
    for(unsigned int j =0; j < childs.size(); ++j)
      {listInds.push_back(linkIndex(x,j));}
    return IndexSet(listInds);
  }
  template IndexSet linkInds<BinaryTree>(BinaryTree const& x, int i);

  template <typename TreeType>
  IndexSet
  linkInds(TreeType const& x)
  {
    auto N = size(x);
    auto inds = IndexSetBuilder(N-1);
    for( auto n : range1(N-1) )
      {
	auto s = linkIndex(x,n);
	inds.nextIndex(std::move(s));
      }
    return inds.build();
  }
  template IndexSet linkInds<BinaryTree>(BinaryTree const& x);

  template <class TreeType>
  bool
  hasQNs(TreeType const& x)
  {
    for(auto i : range(size(x)))
      if(not hasQNs(x(i))) return false;
    return true;
  }
  template bool hasQNs<BinaryTree>(BinaryTree const& x);

  IndexSet
  siteInds(BinaryTree const& x)
  {
    auto N = length(x);
    auto inds=IndexSet();
    for( auto n : range1(N) )
      {
				inds=unionInds(inds,siteIndex(x,n));
      }
    return inds;
  }

  BinaryTree& BinaryTree::
  replaceSiteInds(IndexSet const& sites)
  {
    auto& x = *this;
    auto N = itensor::length(x);
    if( itensor::length(sites)!=N ) Error("In replaceSiteInds(BinaryTree,IndexSet), number of site indices not equal to number of BinaryTree tensors sites");
    auto sx = itensor::siteInds(x);
    if( equals(sx,sites) ) return x;
    for( auto n : range1(N) )
      {
	auto sn = sites(n);
	A_[x.bottomLayer(n)].replaceInds({sx(n)},{sn});
      }
    return x;
  }

  BinaryTree
  replaceSiteInds(BinaryTree x, IndexSet const& sites)
  {
    x.replaceSiteInds(sites);
    return x;
  }

  BinaryTree& BinaryTree::
  replaceLinkInds(IndexSet const& links)
  {
    auto& x = *this;
    auto N = itensor::size(x);
    if( N==1 ) return x;
    if( itensor::length(links)!=N-1 ) Error("In replaceLinkInds(BinaryTree,IndexSet), number of link indices input is not equal to the number of links of the BinaryTree.");
    auto lx = linkInds(x);
    if( equals(lx,links) ) return x;
    for( auto n : range(N) )
      {
		auto index_to_change=x.childrens(n);
	    for (auto & ind_node : index_to_change)
	      {
			A_.at(n).replaceInds({lx(ind_node)},{links(ind_node)});
	      }
		if (n!=0) A_.at(n).replaceInds({lx(n)},{links(n)});
      }
    return x;
  }

  BinaryTree
  replaceLinkInds(BinaryTree x, IndexSet const& links)
  {
    x.replaceLinkInds(links);
    return x;
  }


  IndexSet
  uniqueSiteInds(MPO const& A, BinaryTree const& x)
  {
    auto N = length(x);
    if( N!=length(x) ) Error("In uniqueSiteInds(MPO,BinaryTree), lengths of MPO and BinaryTree do not match");
    return uniqueSiteInds(A,siteInds(x));
  }


  Cplx
  innerC(BinaryTree const& psi,
	 BinaryTree const& phi)
  {
    auto N = size(psi);
    if(N != size(phi)) Error("inner: mismatched size");

    auto psidag = dag(psi);
    // psidag.replaceSiteInds(siteInds(phi));
    psidag.replaceLinkInds(sim(linkInds(psidag)));

    // auto L = phi(N-1) * psidag(N-1);
    // if(N == 0) return eltC(L);
    // for(int i = N-2 ; i >=0; i-- )
    //   L = L * phi(i) * psidag(i);
    // return eltC(L);

    auto N_sites = length(psi);
    auto height = intlog2(N_sites) - 1;
    std::vector<ITensor> psi2(N_sites / 2 + 2);
    for (auto n : range1(N_sites / 2)) {
      psi2[n] = psi(n + N_sites / 2 - 2) * psidag(n + N_sites / 2 - 2);
    }
    for (int i = height - 1; i >= 0; --i) {
      for (auto n : range1(pow2(i))) {
        psi2[n] = psi(n + pow2(i) - 2) * psi2[2 * n - 1] * psi2[2 * n] * psidag(n + pow2(i) - 2);
      }
    }
    return eltC(psi2[1]);
  }

  void
  inner(BinaryTree const& psi, BinaryTree const& phi, Real& re, Real& im)
  {
    auto z = innerC(psi,phi);
    re = real(z);
    im = imag(z);
  }

  Real
  inner(BinaryTree const& psi, BinaryTree const& phi) //Re[<psi|phi>]
  {
    if(isComplex(psi) || isComplex(phi)) Error("Cannot use inner(...) with complex BinaryTree/MPO, use innerC(...) instead");
    Real re, im;
    inner(psi,phi,re,im);
    return re;
  }


  ////<x|A|y>
  //Cplx
  //innerC(BinaryTree const& x,
  //		MPO const& A,
  //       BinaryTree const& y)
  //    {
  //	auto N = length(A);
  //    if( length(y) != N || length(x) != N ) Error("inner: mismatched N");

  //    // Make the indices of |x> and A|y> match
  //    auto sAy = uniqueSiteInds(A,y);
  //    auto xp = replaceSiteInds(x,sAy);

  //    // Dagger x, since it is the ket
  //    auto xdag = dag(xp);
  //    xdag.replaceLinkInds(sim(linkInds(xdag)));

  //    auto L = y(1) * A(1) * xdag(1);

  //    //L *= (A(0) ? A(0)*A(1) : A(1));

  //    for( auto n : range1(2,N) )
  //        L = L * y(n) * A(n) * xdag(n);

  //    // in A(0) and A(N+1). Add this back?
  //    //if(A(N+1)) L *= A(N+1);

  //    return  = eltC(L);


  Cplx
  innerC(BinaryTree const& x,
	 MPO const& A,
	 BinaryTree const& y)
  {
    auto N = length(A);
    if( length(y) != N || length(x) != N ) Error("inner: mismatched size");

    // Make the indices of |x> and A|y> match
    // auto sAy = uniqueSiteInds(A,y);
    // auto xp = replaceSiteInds(x,sAy);

    // Dagger x, since it is the ket
    // auto xdag = dag(xp);
    // xdag.replaceLinkInds(prime(linkInds(xdag)));
    auto xdag = prime(dag(x));
    xdag.replaceLinkInds(sim(linkInds(xdag)));
 //    std::vector<ITensor> inH(x.size()+N);
 //    for(auto i=1; i<= N; ++i)
 //      {
	// inH.at(x.size()+i-1)= A(i);
 //      }
 //    //By decreasing distance, we check if the tensor are contracted, if not we contracted them
 //    for(int d =x.height(); d >= 0; d--) //Max distance is zero as we do not want to contract into the node k
 //      {
	// auto node_d = x.node_list(0,d); // Get the list of node to check, each element is the node and the node towards it is supposed to point out
	// for(unsigned int i=0; i < node_d.size(); i++)
	//   {
	//     auto node=node_d.at(i)[0];
	//     auto direction=node_d.at(i)[1];
	//     auto to_contract=x.othersLinks(node, direction);
	//     inH.at(node)=y(node);
	//     for (auto & conc : to_contract)
	//       {
	// 		inH.at(node)*=inH.at(conc);
	//       }
	//     inH.at(node)*=xdag(node);
	//   }
 //      }
 //    return eltC(inH[0]);

    auto N_sites = length(x);
    auto height = intlog2(N_sites) - 1;
    std::vector<ITensor> yAx(N_sites + 2);
    for (auto n : range1(N_sites)) {
      yAx[n] = A(n);
    }
    for (int i = height; i >= 0; --i) {
      for (auto n : range1(pow2(i))) {
        yAx[n] = y(n + pow2(i) - 2) * yAx[2 * n - 1] * yAx[2 * n] * xdag(n + pow2(i) - 2);
      }
    }
    return eltC(yAx[1]);
  }

  void
  inner(BinaryTree const& psi,MPO const& A,  BinaryTree const& phi, Real& re, Real& im)
  {
    auto z = innerC(psi,A,phi);
    re = real(z);
    im = imag(z);
  }

  Real
  inner(BinaryTree const& psi, MPO const& A, BinaryTree const& phi) //Re[<psi|phi>]
  {
    if(isComplex(psi) || isComplex(phi)) Error("Cannot use inner(...) with complex BinaryTree/MPO, use innerC(...) instead");
    Real re, im;
    inner(psi,A,phi,re,im);
    return re;
  }


  std::ostream&
  operator<<(std::ostream& s, BinaryTree const& M)
  {
    s << "\n";
    for(int i = 0; i < size(M); ++i)
      {
        s << M(i) << "\n";
      }
    return s;
  }


  template <class TreeType>
  TreeType
  removeQNs(TreeType const& psi)
  {
    int N = size(psi);
    TreeType res;
    res = TreeType(length(psi));
    for(int j = 0; j < N; ++j)
      {
        res.ref(j) = removeQNs(psi(j));
	res.setorthoPos(j,psi.orthoPos(j));
      }
    return res;
  }
  template BinaryTree removeQNs<BinaryTree>(BinaryTree const& psi);

  void
  orthPair(ITensor& L, ITensor& R,  Args const& args)
  {
    auto bnd = commonIndex(L,R);
    if(!bnd) return;

    //    if(args.getBool("Verbose",false))
    //        {
    //        Print(inds(L));
    //        }
    // auto original_link_tags = tags(bnd);
    // ITensor A,D,B(bnd);
    // auto spec = svd(L,A,D,B,{args,"LeftTags=",original_link_tags});
    // L = A;
    // R *= (D*B);
    ITensor A,B(bnd);
    qr(L, A, B);
    L = A;
    R *= B;
    auto bnd_qr = commonIndex(L, R);
    L.replaceInds({bnd_qr}, {bnd});
    R.replaceInds({bnd_qr}, {bnd});
  }

} //namespace itensor
