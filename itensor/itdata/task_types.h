//
// Distributed under the ITensor Library License, Version 1.2
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_TASK_TYPES_H_
#define __ITENSOR_TASK_TYPES_H_

#include "itensor/util/infarray.h"
#include "itensor/util/print.h"
#include "itensor/tensor/permute.h"
#include "itensor/real.h"
#include "itensor/indexset.h"

namespace itensor {

//
// Task Types
// 

struct MultReal
    {
    Real r;
    MultReal(Real r_) : r(r_) { }
    };

struct MultCplx
    {
    Cplx z;
    MultCplx(Cplx z_) : z(z_) { }
    };

template<typename IndexT>
struct GetElt
    {
    using Inds = InfArray<long,28ul>;

    const IndexSetT<IndexT>& is;
    const Inds& inds;

    GetElt(const IndexSetT<IndexT>& is_,
           const Inds& inds_);
    };

template<typename T, typename IndexT>
struct SetElt
    {
    using Inds = InfArray<long,28ul>;
    T elt;
    const IndexSetT<IndexT>& is;
    const Inds& inds;

    SetElt(T elt_,
           const IndexSetT<IndexT>& is_,
           const Inds& inds_);
    };


template<typename IndexT>
struct NormNoScale
    {
    const IndexSetT<IndexT>& is;
    NormNoScale(const IndexSetT<IndexT>& is_) : is(is_) { }
    };

template<typename IndexT>
struct PrintIT
    {
    std::ostream& s;
    const LogNumber& x;
    const IndexSetT<IndexT>& is;
    Real scalefac;
    bool print_data;

    PrintIT(std::ostream& s_,
            const LogNumber& x_,
            const IndexSetT<IndexT>& is_,
            bool print_data_)
        : s(s_), x(x_), is(is_), scalefac(1.), print_data(print_data_)
        { 
        if(!x.isTooBigForReal()) scalefac = x.real0();
        }

    template<typename D>
    void
    printInfo(const D& d, 
              std::string type_name,
              Real nrm_no_scale = -1)
        {
        s << "{log(scale)=" << format("%.2f",x.logNum());
        if(nrm_no_scale > 0)
            {
            if(!x.isTooBigForReal()) s << ", norm=";
            else  s << ", norm(omitting large scale)=";
            s << format("%.2f",fabs(scalefac)*nrm_no_scale);
            }
        s << " (" << type_name << ")}\n";
        }

    void
    printVal(double val)
        {
        if(std::fabs(val) > 1E-10)
            s << val << "\n";
        else
            s << format("%.8E\n",val);
        }

    void
    printVal(const Cplx& val)
        {
        if(std::norm(val) > 1E-10)
            {
            auto sgn = (val.imag() < 0 ? '-' : '+');
            s << val.real() << sgn << std::fabs(val.imag()) << "i\n";
            }
        else
            {
            s << format("%.8E\n",val);
            }
        }
    };

struct Conj { };

struct CheckComplex { };

template<typename IndexT>
struct SumEls
    {
    const IndexSetT<IndexT>& is;
    SumEls(const IndexSetT<IndexT>& is_) : is(is_) { }
    };

template<typename F>
struct ApplyIT
    {
    F& f;
    ApplyIT(F&& f_) : f(f_)  { }
    bool constexpr static 
    realToComplex()
        {
        return std::is_same<typename std::result_of<F(Real)>::type,Cplx>::value;
        }
    };

template<typename F, typename T = typename std::result_of<F()>::type>
struct GenerateIT
    {
    F& f;
    GenerateIT(F&& f_) : f(f_)  { }
    };

template <typename F>
struct VisitIT
    {
    F& f;
    Real scale_fac;
    VisitIT(F&& f_, const LogNumber& scale)
        : f(f_), scale_fac(scale.real0())
        { }
    };

struct FillReal
    {
    Real r;
    FillReal(Real r_) : r(r_) { }
    };

struct FillCplx
    {
    Cplx z;
    FillCplx(Cplx z_) : z(z_) { }
    };

struct TakeReal { };
struct TakeImag { };

template<typename IndexT>
struct PlusEQ
    {
    using permutation = Permutation;
    using index_type = IndexT;
    using iset_type = IndexSetT<index_type>;
    private:
    const Permutation *perm_ = nullptr;
    const iset_type *is1_ = nullptr,
                    *is2_ = nullptr;
    public:

    Real fac = NAN;

    PlusEQ(Real fac_) :
        fac(fac_)
        { }

    PlusEQ(const Permutation& P,
           const iset_type& is1,
           const iset_type& is2,
           Real fac_) :
        perm_(&P),
        is1_(&is1),
        is2_(&is2),
        fac(fac_)
        { }

    bool
    hasPerm() const { return bool(perm_); }

    const Permutation&
    perm() const { return *perm_; }

    const iset_type&
    is1() const { return *is1_; }

    const iset_type&
    is2() const { return *is2_; }
    };


template<typename IndexT>
struct Contract
    {
    using index_type = IndexT;
    using iset_type = IndexSetT<IndexT>;
    const Label &Lind,
                &Rind;

    const iset_type &Lis,
                    &Ris;

    iset_type Nis; //new IndexSet
    Real scalefac = NAN;

    Contract(const iset_type& Lis_,
             const Label& Lind_,
             const iset_type& Ris_,
             const Label& Rind_) :
        Lind(Lind_),
        Rind(Rind_),
        Lis(Lis_),
        Ris(Ris_)
        { }

    Contract(const Contract& other) = delete;
    Contract& operator=(const Contract& other) = delete;

    Contract(Contract&& other):
        Lind(other.Lind),
        Rind(other.Rind),
        Lis(other.Lis),
        Ris(other.Ris),
        Nis(std::move(other.Nis)),
        scalefac(other.scalefac)
        { }

    template<typename Data>
    void
    computeScalefac(Data& dat)
        {
        scalefac = 0;
        for(auto elt : dat) scalefac += elt*elt;
        scalefac = std::sqrt(scalefac);
        if(scalefac == 0) return;
        for(auto& elt : dat) elt /= scalefac;
        }
    };

enum class 
StorageType
    { 
    Null=0, 
    ITReal=1, 
    ITCplx=2, 
    ITCombiner=3, 
    ITDiagReal=4, 
    ITDiagCplx=5,
    IQTData=6
    }; 

struct Write
    {
    std::ostream& s;

    Write(std::ostream& s_) : s(s_) { }

    template<class T>
    void
    writeType(StorageType type, const T& data)
        {
        s.write((char*)&type,sizeof(type));
        write(s,data); 
        }
    };


namespace detail {

template<typename I>
void
checkEltInd(const IndexSetT<I>& is,
            const typename GetElt<I>::Inds& inds)
    {
    for(size_t k = 0; k < inds.size(); ++k)
        {
        auto i = inds[k];
        if(i < 0)
            {
            print("inds = ");
            for(auto j : inds) print(1+j," ");
            println();
            Error("Out of range: IndexVals/IQIndexVals are 1-indexed for getting tensor elements");
            }
        if(i >= is[k].m())
            {
            print("inds = ");
            for(auto j : inds) print(1+j," ");
            println();
            Error(format("Out of range: IndexVal/IQIndexVal at position %d has val %d > %s",1+k,1+i,Index(is[k])));
            }
        }
    }

} //namespace detail

template<typename IndexT>
GetElt<IndexT>::
GetElt(const IndexSetT<IndexT>& is_,
       const Inds& inds_)
  : is(is_),
    inds(inds_)
    { 
#ifdef DEBUG
    detail::checkEltInd(is,inds);
#endif
    }

template<typename T, typename IndexT>
SetElt<T,IndexT>::
SetElt(T elt_,
       const IndexSetT<IndexT>& is_,
       const Inds& inds_)
    : elt(elt_), is(is_), inds(inds_)
    { 
#ifdef DEBUG
    detail::checkEltInd(is,inds);
#endif
    }

} //namespace itensor 

#endif