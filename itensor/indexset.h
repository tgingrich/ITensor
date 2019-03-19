//
// Distributed under the ITensor Library License, Version 1.2
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_INDEXSET_H
#define __ITENSOR_INDEXSET_H
#include <algorithm>
#include "itensor/util/safe_ptr.h"
#include "itensor/index.h"
#include "itensor/tensor/contract.h"
#include "itensor/tensor/range.h"
#include "itensor/tensor/types.h"
#include "itensor/tensor/permutation.h"

namespace itensor {


class IndexSet;

using IndexSetBuilder = RangeBuilderT<IndexSet>;

template<typename I>
class IndexSetIter;

//
// IndexSet
//
// When constructed from a collection of indices,
// (as an explicit set of arguments or via
// a container) puts the indices with m>1 to the
// front and those with m==1 at the back but otherwise
// keeps the indices in the order given.
//

void
checkQNConsistent(IndexSet const&);

class IndexSet : public RangeT<Index>
    {
    public:
    using extent_type = index_type;
    using range_type = RangeT<Index>;
    using parent = RangeT<Index>;
    using size_type = typename range_type::size_type;
    using storage_type = typename range_type::storage_type;
    using value_type = Index;
    using iterator = IndexSetIter<Index>;
    using const_iterator = IndexSetIter<const Index>;

    public:

    IndexSet() { }

    // construct from 1 or more indices
    template <typename... Inds>
    explicit
    IndexSet(Index const& i1, 
             Inds&&... inds)
      : parent(i1,std::forward<Inds>(inds)...)
        { 
        checkQNConsistent(*this);
        }

    IndexSet(std::initializer_list<Index> const& ii)
      : parent(ii) 
        { 
        checkQNConsistent(*this);
        }

    IndexSet(std::vector<Index> const& ii)
      : parent(ii) 
        { 
        checkQNConsistent(*this);
        }

    template<size_t N>
    IndexSet(std::array<Index,N> const& ii)
      : parent(ii)
        {
        checkQNConsistent(*this);
        }

    template<typename IndxContainer>
    explicit
    IndexSet(IndxContainer && ii) 
      : parent(std::forward<IndxContainer>(ii)) 
        { 
        checkQNConsistent(*this);
        }

    explicit
    IndexSet(storage_type && store) 
      : parent(std::move(store)) 
        { 
        checkQNConsistent(*this);
        }

    explicit operator bool() const { return !parent::empty(); }

    long
    extent(size_type i) const { return parent::extent(i); }

    size_type
    stride(size_type i) const { return parent::stride(i); }

    long
    order() const { return parent::order(); }
    
    // 0-indexed access
    Index &
    operator[](size_type i)
        { 
#ifdef DEBUG
        if(i >= parent::size()) throw ITError("IndexSet[i] arg out of range");
#endif
        return parent::index(i);
        }

    // 1-indexed access
    Index &
    index(size_type I)
        { 
#ifdef DEBUG
        if(I < 1 || I > parent::size()) throw ITError("IndexSet.index(i) arg out of range");
#endif
        return operator[](I-1);
        }

    // 0-indexed access
    Index const&
    operator[](size_type i) const
        { 
#ifdef DEBUG
        if(i >= parent::size()) throw ITError("IndexSet[i] arg out of range");
#endif
        return parent::index(i);
        }

    // 1-indexed access
    Index const&
    index(size_type I) const
        { 
#ifdef DEBUG
        if(I < 1 || I > parent::size()) throw ITError("IndexSet.index(i) arg out of range");
#endif
        return operator[](I-1);
        }

    parent const&
    range() const { return *this; }

    void
    dag();

    void
    swap(IndexSet & other) { parent::swap(other); }

    Index const&
    front() const { return parent::front().ind; }

    Index const&
    back() const { return parent::back().ind; }

    iterator
    begin();

    iterator
    end();

    const_iterator
    begin() const;

    const_iterator
    end() const;

    const_iterator
    cbegin() const;

    const_iterator
    cend() const;

    //
    // Tag methods
    //

    void
    setTags(TagSet const& tsnew);

    void
    setTags(TagSet const& tsnew, 
            IndexSet const& ismatch);

    template<typename... VarArgs>
    void
    setTags(TagSet const& tsnew,
            Index const& imatch1,
            VarArgs&&... vargs)
      {
      setTags(tsnew,IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
      }

    void
    setTags(TagSet const& tsnew, 
            TagSet const& tsmatch);

    void
    addTags(TagSet const& tsadd);

    void
    addTags(TagSet const& tsadd, 
            IndexSet const& ismatch);

    template<typename... VarArgs>
    void
    addTags(TagSet const& tsadd,
            Index const& imatch1,
            VarArgs&&... vargs)
      {
      addTags(tsadd,IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
      }

    void
    addTags(TagSet const& tsadd,
            TagSet const& tsmatch);

    void
    removeTags(TagSet const& tsremove);

    void
    removeTags(TagSet const& tsremove, 
               IndexSet const& ismatch);

    template<typename... VarArgs>
    void
    removeTags(TagSet const& tsremove,
               Index const& imatch1,
               VarArgs&&... vargs)
      {
      removeTags(tsremove,IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
      }

    void
    removeTags(TagSet const& tsremove, 
               TagSet const& tsmatch);

    void
    replaceTags(TagSet const& tsold, 
                TagSet const& tsnew);

    void
    replaceTags(TagSet const& tsold, 
                TagSet const& tsnew, 
                IndexSet const& ismatch);

    template<typename... VarArgs>
    void
    replaceTags(TagSet const& tsold,
                TagSet const& tsnew,
                Index const& imatch1,
                VarArgs&&... vargs)
      {
      replaceTags(tsold,tsnew,IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
      }

    void
    replaceTags(TagSet const& tsold, 
                TagSet const& tsnew,
                TagSet const& tsmatch);

    template<typename... VarArgs>
    void
    swapTags(TagSet const& ts1,
             TagSet const& ts2,
             VarArgs&&... vargs);

    //
    // Integer tag convenience functions
    //

    //
    // Set the integer tag of indices to plnew
    //

    void
    setPrime(int plnew);

    void
    setPrime(int plnew,
             IndexSet const& ismatch);

    template<typename... VarArgs>
    void
    setPrime(int plnew,
             Index const& imatch1,
             VarArgs&&... vargs)
        {
        setPrime(plnew,IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
        }

    void
    setPrime(int plnew,
             TagSet const& tsmatch);

    template<typename... VarArgs>
    void
    noPrime(VarArgs&&... vargs)
        {
        setPrime(0,std::forward<VarArgs>(vargs)...);
        }

    //
    // Increase the integer tag of indices by plinc
    //

    void
    prime(int plinc);

    void
    prime()
      {
      prime(1);
      }

    void
    prime(int plinc,
          IndexSet const& ismatch);

    void
    prime(IndexSet const& ismatch)
      {
      prime(1,ismatch);
      }

    template<typename... VarArgs>
    void
    prime(int plinc,
          Index const& imatch1,
          VarArgs&&... vargs)
      {
      prime(plinc,IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
      }

    template<typename... VarArgs>
    void
    prime(Index const& imatch1,
          VarArgs&&... vargs)
      {
      prime(IndexSet(imatch1,std::forward<VarArgs>(vargs)...));
      }

    void
    prime(int plinc,
          TagSet const& tsmatch);

    void
    prime(TagSet const& tsmatch)
      {
      prime(1,tsmatch);
      }
 
    // Remove QNs from all indices in the IndexSet
    void
    removeQNs();

    //
    // Deprecated
    //

    long
    r() const { return this->order(); }
    
    void
    prime(Index const& imatch,
          int plinc)
        {
        Error("Error: .prime(Index,int) is no longer supported, use .prime(int,Index) instead.");
        }

    };

void
read(std::istream& s, IndexSet & is);

void
write(std::ostream& s, IndexSet const& is);

auto inline
rangeBegin(IndexSet const& is) -> decltype(is.range().begin())
    {
    return is.range().begin();
    }

auto inline
rangeEnd(IndexSet const& is) -> decltype(is.range().end())
    {
    return is.range().end();
    }

// Find the Index containing tags in the specified TagSet 
// and matching the specified prime level
// This is useful if we know there is an Index
// that contains Tags tsmatch, but don't know the other tags
// If multiple indices are found, throw an error
// If none are found, return a default Index
Index
findIndex(IndexSet const& is,
          TagSet const& tsmatch);

//
//
// IndexSet Primelevel Methods
//


//Replace all indices with 'similar' indices 
//with the same properties but which don't compare equal 
//to the indices they replace (using sim(Index) function)
IndexSet
sim(IndexSet is);
IndexSet
sim(IndexSet is, 
    IndexSet const& ismatch);
IndexSet
sim(IndexSet is, 
    TagSet const& tsmatch);


//
// IndexSet helper methods
//


//
// Given IndexSet iset and Index I,
// return int j such that iset[j] == I.
// If not found, returns -1
//
int
indexPosition(IndexSet const& is, 
              Index const& imatch);

std::vector<int>
indexPositions(IndexSet const& is,
               IndexSet const& ismatch);

Arrow
dir(IndexSet const& is, Index const& I);


bool
hasIndex(IndexSet const& iset, 
         Index const& I);

long
minDim(IndexSet const& iset);

long
maxDim(IndexSet const& iset);

void
contractIS(IndexSet const& Lis,
           IndexSet const& Ris,
           IndexSet & Nis,
           bool sortResult = false);

template<class LabelT>
void
contractIS(IndexSet const& Lis,
           LabelT const& Lind,
           IndexSet const& Ris,
           LabelT const& Rind,
           IndexSet & Nis,
           LabelT & Nind,
           bool sortResult = false);

template<class LabelT>
void
ncprod(IndexSet const& Lis,
       LabelT const& Lind,
       IndexSet const& Ris,
       LabelT const& Rind,
       IndexSet & Nis,
       LabelT & Nind);

std::ostream&
operator<<(std::ostream& s, IndexSet const& is);

template<typename index_type_>
class IndexSetIter
    { 
    public:
    using index_type = stdx::remove_const_t<index_type_>;
    using value_type = index_type;
    using reference = index_type_&;
    using difference_type = std::ptrdiff_t;
    using pointer = index_type_*;
    using iterator_category = std::random_access_iterator_tag;
    using indexset_type = stdx::conditional_t<std::is_const<index_type_>::value,
                                             const IndexSet,
                                             IndexSet>;
    using range_ptr = typename RangeT<index_type>::value_type*;
    using const_range_ptr = const typename RangeT<index_type>::value_type*;
    using data_ptr = stdx::conditional_t<std::is_const<index_type_>::value,
                                      const_range_ptr,
                                      range_ptr>;
    private:
    size_t off_ = 0;
    indexset_type* p_; 
    public: 

    IndexSetIter() : p_(nullptr) { }

    explicit
    IndexSetIter(indexset_type & is) : p_(&is) { }

    size_t
    offset() const { return off_; }

    IndexSetIter& 
    operator++() 
        { 
        ++off_; 
        return *this; 
        } 

    IndexSetIter 
    operator++(int) 
        { 
        auto tmp = *this; //save copy of this
        ++off_; 
        return tmp; 
        } 

    IndexSetIter& 
    operator+=(difference_type x) 
        { 
        off_ += x;
        return *this; 
        } 

    IndexSetIter& 
    operator--( ) 
        { 
        --off_;
        return *this; 
        } 

    IndexSetIter 
    operator--(int) 
        { 
        auto tmp = *this; //save copy of this
        --off_;
        return tmp; 
        } 

    IndexSetIter& 
    operator-=(difference_type x) 
        { 
        off_ -= x;
        return *this; 
        } 

    reference 
    operator[](difference_type n) { return p_->operator[](n); } 

    reference 
    operator*() { return p_->operator[](off_); }  

    pointer 
    operator->() { return &(p_->operator[](off_)); }

    IndexSetIter static
    makeEnd(indexset_type & is)
        {
        IndexSetIter end;
        end.p_ = &is;
        end.off_ = is.size();
        return end;
        }
    }; 

template <typename T>
bool 
operator==(const IndexSetIter<T>& x, const IndexSetIter<T>& y) 
    { 
    return x.offset() == y.offset(); 
    } 

template <typename T>
bool 
operator!=(const IndexSetIter<T>& x, const IndexSetIter<T>& y) 
    { 
    return x.offset() != y.offset(); 
    } 

template <typename T>
bool 
operator<(const IndexSetIter<T>& x, const IndexSetIter<T>& y) 
    { 
    return x.offset() < y.offset(); 
    } 

template <typename T>
IndexSetIter<T>
operator+(IndexSetIter<T> x, 
          typename IndexSetIter<T>::difference_type d) 
    { 
    return x += d;
    } 

template <typename T>
IndexSetIter<T>
operator+(typename IndexSetIter<T>::difference_type d, 
          IndexSetIter<T> x) 
    { 
    return x += d;
    } 

bool
hasQNs(IndexSet const& is);

void
checkIndexSet(IndexSet const& is);

void
checkIndexPositions(std::vector<int> const& is);

IndexSet
unionInds(IndexSet const& is1,
          IndexSet const& is2);

} //namespace itensor

#include "itensor/indexset_impl.h"

#endif
