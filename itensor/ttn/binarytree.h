//The following ifndef/define/endif pattern is called a
//scope guard, and prevents the C++ compiler (actually, preprocessor)
//from including a header file more than once.
#ifndef __BINARY_TREE_H_
#define __BINARY_TREE_H_

#include <string>
#include <iostream>
#include <climits>
#include "itensor/decomp.h"
#include "itensor/mps/siteset.h"
#include "itensor/mps/mps.h"
#include "itensor/mps/mpo.h"

//Integer log2 function

inline unsigned int intlog2 (unsigned int val) {
  if (val == 0) return UINT_MAX;
  if (val == 1) return 0;
  unsigned int ret = 0;
  while (val > 1) {
    val >>= 1;
    ret++;
  }
  return ret;
}

inline bool islog2(int num) { return ((num != 0) && ((num &(num - 1)) == 0));}

inline constexpr int pow2 (unsigned int i)
{
  return std::uint64_t(1) << i;
}

namespace itensor {

  class InitState;

  /* Ordering of the index are horizontal travel
  // The root node is 0 the two below are 1 and 2
  0
  / \
  1  2
  /\  /\
  3 4 5 6
  ...

  The edges are labeled by the node below the edge (hence there is no edge 0)
  */

  class BinaryTree
  {
  protected:
    int N_;
    int height_;
    int N_sites_;

    mutable    std::vector<ITensor> A_; // List of ITensor inside the tree
    std::vector<int> orth_pos_; //We store the direction of orthogonality for each node (ie the node towards ), set to -1 if not orthogonalized

    std::vector<int> order_; //We store the direction of orthogonality for each node (ie the node towards ), set to -1 if not orthogonalized
    std::vector<int> reverse_order_; //We store the direction of orthogonality for each node (ie the node towards ), set to -1 if not orthogonalized
    int order_start_;
    int site_dim_;
  public:

    //
    // BinaryTree Constructors
    //

    BinaryTree();

    BinaryTree(int length);

    BinaryTree(SiteSet const& sites, int m = 1);

    BinaryTree(IndexSet const& sites, int m = 1);

    BinaryTree(InitState const& initState);

    BinaryTree(BinaryTree const& other);

    BinaryTree&
    operator=(BinaryTree const& other);

    ~BinaryTree();

    void
    init_tensors(const InitState& initState);


    //
    // Binary Tree helper
    //
    int parent(int i) const {
      if (i== 0) return -1;//Check if we are not at the top
      //if (i > N_) return -1;// We are out of the tree there is no parents
      return (i-1)/2;
    }

    int leftchild(int i) const {
      if ((2*i+1) > N_-1) return -1;//Check if we are below the bottom
      return 2*i+1;
    }

    int rightchild(int i) const {
      if ((2*i+2) > N_-1) return -1;//Check if we are below the bottom
      return 2*i+2;
    }

    std::vector<int> childrens(int i) const {
      if (((2*i+1) > N_-1) || i < 0) return std::vector<int>(0); // Out of the tree
      return std::vector<int>{ this->leftchild(i),this->rightchild(i) };
    }

    int sibling(int i) const { // Return the id of the other children of the parent node
      if( 2*((i-1)/2)+1 == i ) return i+1; // If come back to the same children we are the 2*i+1 otherwise we are the 2*i+2
      return i-1;
    }

    std::vector<int> othersLinks(int i, int dir) const{ // Given a node and a direction give all other links to the node

      if (dir==parent(i)) return std::vector<int>({2*i+1,2*i+2});
      else if (i == 0) return std::vector<int>({sibling(dir)});
      else return std::vector<int>({parent(i),sibling(dir)});

    }

    int depth(int i) const {return intlog2(i+1);} // Find depth of the node (i.e. distance from root node)

    int  //Return the id of the tensor connected to the site i
    bottomLayer(int site) const
    {
      if(site <= 0)  site=N_sites_;
      return pow2(height_)-1+(site-1)/2;
    }

    std::vector<int> children(int i,int distance) const; // Return all the children of the node that are at a certain distance, there is 2^distance of them

    std::vector<int[2]> node_list (int i, int distance) const; // Return all the nodes that are at a certain distance from a given node

    //
    // BinaryTree Accessor Methods
    //

    int size() const { return N_;}

    int length() const { return N_sites_;}

    int height() const { return height_;}

    int site_dim() const { return site_dim_; }

    std::vector<int> orthoVect() const { return orth_pos_;}

    int orthoPos(int i) const {return orth_pos_.at(i);}

    void setorthoPos(int i, int val) { orth_pos_.at(i) = val;}

		void setOrthoLink(int i, int j)
		{
			orth_pos_.at(i) = j;// We update the orthogonalisation memory
			orth_pos_.at(j) = -1; // The next one is not any more orthogonal

		}

    void setOrder(Args const& args = Args::global());

    void setOrder(std::vector<int> new_order);

    void sweepnext(int &b, int &ha,Args const& args = Args::global()); //Travel method

    int startPoint(Args const& args = Args::global()) const;

    explicit operator bool() const { return (not A_.empty()); }

    bool
    doWrite() const { return false; }
    void
    doWrite(bool val,
            Args const& args = Args::global())
    {
      Error("doWrite is not supported by BinaryTree");
    }

    // Read-only access to i'th BinaryTree tensor
    ITensor const&
    operator()(int i) const;

    //Returns reference to i'th BinaryTree tensor
    //which allows reading and writing
    ITensor&
    ref(int i);

    void
    set(int i, ITensor const& nA) {
      ref(i) = nA;
      orth_pos_[i]=-1; // The site is not longer orthogonalized
    }

    void
    set(int i, ITensor && nA) {
      ref(i) = std::move(nA);
    }

    // Dagger all BinaryTree tensors
    BinaryTree&
    dag()
    {
      for(auto i : range(N_))
	A_[i].dag();
      return *this;
    }

    Real normalize();

    BinaryTree&
    replaceSiteInds(IndexSet const& sites);

    BinaryTree&
    replaceLinkInds(IndexSet const& links);



    //
    //BinaryTree Index Methods
    //

    BinaryTree&
    setTags(TagSet const& ts, IndexSet const& is)
    {
      for(int i=0; i < N_ ; ++i)
	A_[i].setTags(ts,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    setTags(VarArgs&&... vargs)
    {
      for(int i=0; i < N_ ; ++i)
	A_[i].setTags(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    noTags(IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].noTags(is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    noTags(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].noTags(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    addTags(TagSet const& ts, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].addTags(ts,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    addTags(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].addTags(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    removeTags(TagSet const& ts, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].removeTags(ts,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    removeTags(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].removeTags(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    replaceTags(TagSet const& ts1, TagSet const& ts2, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].replaceTags(ts1,ts2,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    replaceTags(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].replaceTags(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    swapTags(TagSet const& ts1, TagSet const& ts2, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].swapTags(ts1,ts2,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    swapTags(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].swapTags(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    prime(int plev, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].prime(plev,is);
      return *this;
    }

    BinaryTree&
    prime(IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].prime(is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    prime(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].prime(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    setPrime(int plev, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].setPrime(plev,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    setPrime(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].setPrime(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    mapPrime(int plevold, int plevnew, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].mapPrime(plevold,plevnew,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    mapPrime(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].mapPrime(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    swapPrime(int plevold, int plevnew, IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].swapPrime(plevold,plevnew,is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    swapPrime(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].swapPrime(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    BinaryTree&
    noPrime(IndexSet const& is)
    {
      for(int i=0; i < N_; ++i)
	A_[i].noPrime(is);
      return *this;
    }

    template<typename... VarArgs>
    BinaryTree&
    noPrime(VarArgs&&... vargs)
    {
      for(int i=0; i < N_; ++i)
	A_[i].noPrime(std::forward<VarArgs>(vargs)...);
      return *this;
    }

    // Randomize the tensors of the BinaryTree
    // Right now, only supports randomizing dim = 1 BinaryTree
    BinaryTree&
    randomize(Args const& args = Args::global());

    Spectrum
    svdBond(int b1,
            ITensor const& AA,
            int b2,
            Args args = Args::global());

    template<class LocalOpT>
    Spectrum
    svdBond(int b1,
            ITensor const& AA,
            int b2,
            LocalOpT const& PH,
            Args args = Args::global());

    //Move the orthogonality center to site i
    BinaryTree&
    position(int i, Args args = Args::global());

    BinaryTree&
    orthogonalize(Args args = Args::global());

  }; //class BinaryTree

  BinaryTree&
  operator*=(BinaryTree & x, Real a);

  BinaryTree&
  operator/=(BinaryTree & x, Real a);

  BinaryTree
  operator*(BinaryTree x, Real r);

  BinaryTree
  operator*(Real r, BinaryTree x);

  BinaryTree&
  operator*=(BinaryTree & x, Cplx z);

  BinaryTree&
  operator/=(BinaryTree & x, Cplx z);

  BinaryTree
  operator*(BinaryTree x, Cplx z);

  BinaryTree
  operator*(Cplx z, BinaryTree x);

  BinaryTree
  dag(BinaryTree A);

  //
  // BinaryTree tag functions
  //

  BinaryTree
  setTags(BinaryTree A, TagSet const& ts, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  setTags(BinaryTree A,
	  VarArgs&&... vargs)
  {
    A.setTags(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  noTags(BinaryTree A, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  noTags(BinaryTree A,
	 VarArgs&&... vargs)
  {
    A.noTags(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  addTags(BinaryTree A, TagSet const& ts, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  addTags(BinaryTree A,
	  VarArgs&&... vargs)
  {
    A.addTags(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  removeTags(BinaryTree A, TagSet const& ts, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  removeTags(BinaryTree A,
	     VarArgs&&... vargs)
  {
    A.removeTags(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  replaceTags(BinaryTree A, TagSet const& ts1, TagSet const& ts2, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  replaceTags(BinaryTree A,
	      VarArgs&&... vargs)
  {
    A.replaceTags(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  swapTags(BinaryTree A, TagSet const& ts1, TagSet const& ts2, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  swapTags(BinaryTree A,
	   VarArgs&&... vargs)
  {
    A.swapTags(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  prime(BinaryTree A, int plev, IndexSet const& is);

  BinaryTree
  prime(BinaryTree A, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  prime(BinaryTree A,
	VarArgs&&... vargs)
  {
    A.prime(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  setPrime(BinaryTree A, int plev, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  setPrime(BinaryTree A,
	   VarArgs&&... vargs)
  {
    A.setPrime(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  mapPrime(BinaryTree A, int plevold, int plevnew, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  mapPrime(BinaryTree A,
	   VarArgs&&... vargs)
  {
    A.mapPrime(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  swapPrime(BinaryTree A, int plevold, int plevnew, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  swapPrime(BinaryTree A,
	    VarArgs&&... vargs)
  {
    A.swapPrime(std::forward<VarArgs>(vargs)...);
    return A;
  }

  BinaryTree
  noPrime(BinaryTree A, IndexSet const& is);

  template<typename... VarArgs>
  BinaryTree
  noPrime(BinaryTree A,
	  VarArgs&&... vargs)
  {
    A.noPrime(std::forward<VarArgs>(vargs)...);
    return A;
  }


  //
  // Other Methods Related to BinaryTree
  //

  int
  length(BinaryTree const& W);

  int size(BinaryTree const& W);

  template <class BinaryTreeType>
  bool
  hasQNs(BinaryTreeType const& x);


  //Create a random BinaryTree
  BinaryTree
  randomBinaryTree(SiteSet const& sites,
            int m,
            Args const& args = Args::global());

  BinaryTree
  randomBinaryTree(InitState const& initstate,
            int m,
            Args const& args = Args::global());

  BinaryTree
  randomBinaryTree(InitState const& initstate,
            Args const& args = Args::global());

  //Remove the QNs of each tensor of the BinaryTree
  template <class BinaryTreeType>
  BinaryTreeType
  removeQNs(BinaryTreeType const& x);


  bool
  isComplex(BinaryTree const& x);

  bool
  isOrtho(BinaryTree const& x);

  int
  orthoCenter(BinaryTree const& x);

	int
  findCenter(BinaryTree const& x);

  Real
  norm(BinaryTree const& x);

  bool
  hasSiteInds(BinaryTree const& x, IndexSet const& sites);

  template <class BinaryTreeType>
  IndexSet
  siteInds(BinaryTreeType const& W, int b);

  IndexSet
  siteInds(BinaryTree const& x);

  BinaryTree
  replaceSiteInds(BinaryTree x, IndexSet const& sites);

  BinaryTree
  replaceLinkInds(BinaryTree x, IndexSet const& links);

  Index
  siteIndex(BinaryTree const& x, int site);

  template<typename BinaryTreeType>
  Index
  linkIndex(BinaryTreeType const& x, int b);

  template<typename BinaryTreeType>
  IndexSet
  linkInds(BinaryTreeType const& x, int b);

  template<typename BinaryTreeType>
  IndexSet
  linkInds(BinaryTreeType const& x);

  template<typename BinaryTreeType>
  Real
  averageLinkDim(BinaryTreeType const& x);

  template<typename BinaryTreeType>
  int
  maxLinkDim(BinaryTreeType const& x);

  bool
  checkQNs(BinaryTree const& x);

  QN
  totalQN(BinaryTree const& x);

  // Get the site Indices that are unique to A
  IndexSet
  uniqueSiteInds(MPO const& A, BinaryTree const& x);


  // Re[<x|y>]
  Real
  inner(BinaryTree const& x, BinaryTree const& y);

  // <x|y>
  Cplx
  innerC(BinaryTree const& x,
	 BinaryTree const& y);

  // <x|y>
  void
  inner(BinaryTree const& x,
	BinaryTree const& y,
	Real& re, Real& im);

  //Inner product with MPO
  // Re[<x|y>]
  Real
  inner(BinaryTree const& x, MPO const& A,  BinaryTree const& y);

  // <x|y>
  Cplx
  innerC(BinaryTree const& x,
	 MPO const& A,
	 BinaryTree const& y);

  // <x|y>
  void
  inner(BinaryTree const& x,
	MPO const& A,
	BinaryTree const& y,
	Real& re, Real& im);

  std::ostream&
  operator<<(std::ostream& s, BinaryTree const& M);

  void
  orthPair(ITensor& A1, ITensor& A2, Args const& args);


  template <class BinaryTreeType>
  void inline
  sweepnext(int &b, int &ha, BinaryTreeType &Tree,  Args const& args = Args::global())
  {
    Tree.sweepnext(b,ha,args);
  }


  template <class BinaryTreeType>
  void inline
  sweepnext(int &b, int &ha, BinaryTreeType &Tree)
  {
    auto args = Args("NumCenter=",1);
    Tree.sweepnext(b,ha,args);
  }

  template <class BinaryTreeType>
  void inline
  sweepnext2(int &b, int &ha, BinaryTreeType &Tree)
  {
    auto args = Args("NumCenter=",2);
    Tree.sweepnext(b,ha,args);
  }

} //namespace itensor

#include "binarytree_impl.h"

#endif //__BINARY_TREE_H_
