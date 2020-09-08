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
#ifndef __ITENSOR_LOCAL_OP_TREE
#define __ITENSOR_LOCAL_OP_TREE
#include "itensor/itensor.h"
//#include "itensor/util/print_macro.h"

namespace itensor {

//
// The LocalOpTree class represents
// an MPO or other operator that
// has been projected into the
// reduced Hilbert space of 
// two sites of an BinaryTree.
//
//   .-              -.
//   |    |      |    |
//   L - Op1 -- Op2 - R
//   |    |      |    |
//   '-              -'
//
// (Note that L, Op1, Op2 and R
//  are not required to have this
//  precise structure. L and R
//  can even be null in which case
//  they will not be used.)
//


class LocalOpTree
    {
    ITensor const* Op1_;
    ITensor const* Op2_;
    ITensor const* L_;
    ITensor const* R_;
    mutable size_t size_;
    int nc_;
    public:


    //
    // Constructors
    //

    LocalOpTree(Args const& args = Args::global());

    LocalOpTree(ITensor const& Op1,
            Args const& args = Args::global());

    LocalOpTree(ITensor const& Op1, 
            ITensor const& Op2,
            Args const& args = Args::global());
  
    LocalOpTree(ITensor const& Op1, 
            ITensor const& L, 
            ITensor const& R,
            Args const& args = Args::global());

    LocalOpTree(ITensor const& Op1, 
            ITensor const& Op2, 
            ITensor const& L, 
            ITensor const& R,
            Args const& args = Args::global());

    //
    // Sparse Matrix Methods
    //

    void
    product(ITensor const& phi, ITensor & phip) const;

    Real
    expect(ITensor const& phi) const;

    ITensor
    deltaRho(ITensor const& rho, 
             ITensor const& combine, 
             Direction dir) const;

    ITensor
    diag() const;

    size_t
    size() const;

    int
    numCenter() const { return nc_; }
    void
    numCenter(int val)
        {
        if(val < 0 || val > 2) Error("numCenter must be set to be 0 or 1 or 2");
        nc_ = val;
        }

    //
    // Accessor Methods
    //
    
    void
    updateOp(ITensor const& Op1);

    void
    updateOp(ITensor const& Op1, ITensor const& Op2);
    
    void
    update(ITensor const& L, 
           ITensor const& R);

    void
    update(ITensor const& Op1, 
           ITensor const& L, 
           ITensor const& R);

    void
    update(ITensor const& Op1, 
           ITensor const& Op2, 
           ITensor const& L, 
           ITensor const& R);

    ITensor const&
    Op1() const 
        { 
        if(!(*this)) Error("LocalOpTree is default constructed");
        return *Op1_;
        }

    ITensor const&
    Op2() const 
        { 
        if(!(*this)) Error("LocalOpTree is default constructed");
        return *Op2_;
        }

    ITensor const&
    L() const 
        { 
        if(!(*this)) Error("LocalOpTree is default constructed");
        return *L_;
        }

    ITensor const&
    R() const 
        { 
        if(!(*this)) Error("LocalOpTree is default constructed");
        return *R_;
        }

    explicit operator bool() const 
        {
        return !LIsNull() || !RIsNull();
        }

    bool
    LIsNull() const;

    bool
    RIsNull() const;

    };

inline LocalOpTree::
LocalOpTree(Args const& args)
    :
    Op1_(nullptr),
    Op2_(nullptr),
    L_(nullptr),
    R_(nullptr),
    size_(-1)
    {
    nc_ = args.getInt("NumCenter",2);
    }

inline LocalOpTree::
LocalOpTree(const ITensor& Op1,
        const Args& args)
    : 
    Op1_(nullptr),
    Op2_(nullptr),
    L_(nullptr),
    R_(nullptr),
    size_(-1)
    {
    nc_ = args.getInt("NumCenter",2);
    if(nc_ == 1)	
      updateOp(Op1);
    else
      Error("In LocalOpTree(ITensor), NumCenter cannot be set other than 1");
    }

inline LocalOpTree::
LocalOpTree(const ITensor& Op1, const ITensor& Op2,
        const Args& args)
    : 
    Op1_(nullptr),
    Op2_(nullptr),
    L_(nullptr),
    R_(nullptr),
    size_(-1)
    {
    nc_ = args.getInt("NumCenter",2);
    if(nc_ == 2)
      updateOp(Op1,Op2);
    else if(nc_ == 0)
      update(Op1,Op2);// L, R
    else
      Error("In LocalOpTree(ITensor,ITensor), NumCenter cannot be set other than 2 or 0");
    }

inline LocalOpTree::
LocalOpTree(const ITensor& Op1,
        const ITensor& L, const ITensor& R,
        const Args& args)
    : 
    Op1_(nullptr),
    Op2_(nullptr),
    L_(nullptr),
    R_(nullptr),
    size_(-1)
    {
    nc_ = args.getInt("NumCenter",2);
    if(nc_ == 1)
      update(Op1,L,R);
    else
      Error("In LocalOpTree(ITensor,ITensor,ITensor), NumCenter cannot be set other than 1");
    }

inline LocalOpTree::
LocalOpTree(const ITensor& Op1, const ITensor& Op2, 
        const ITensor& L, const ITensor& R,
        const Args& args)
    : 
    Op1_(nullptr),
    Op2_(nullptr),
    L_(nullptr),
    R_(nullptr),
    size_(-1)
    {
    nc_ = args.getInt("NumCenter",2);
    if(nc_ == 2)
      update(Op1,Op2,L,R);
    else
      Error("In LocalOpTree(ITensor,ITensor,ITensor,ITensor), NumCenter cannot be set other than 2");
    }

void inline LocalOpTree::
updateOp(const ITensor& Op1)
    {
    Op1_ = &Op1;
    Op2_ = nullptr;
    L_ = nullptr;
    R_ = nullptr;
    size_ = -1;
    nc_ = 1;
    }

void inline LocalOpTree::
updateOp(const ITensor& Op1, const ITensor& Op2)
    {
    Op1_ = &Op1;
    Op2_ = &Op2;
    L_ = nullptr;
    R_ = nullptr;
    size_ = -1;
    nc_ = 2;
    }

void inline LocalOpTree::
update(const ITensor& L, const ITensor& R)
    {
    Op1_ = nullptr;
    Op2_ = nullptr;
    L_ = &L;
    R_ = &R;
    size_ = -1;
    nc_ = 0;
    }

void inline LocalOpTree::
update(const ITensor& Op1, 
       const ITensor& L, const ITensor& R)
    {
    updateOp(Op1);
    L_ = &L;
    R_ = &R;
    }

void inline LocalOpTree::
update(const ITensor& Op1, const ITensor& Op2, 
       const ITensor& L, const ITensor& R)
    {
    updateOp(Op1,Op2);
    L_ = &L;
    R_ = &R;
    }

bool inline LocalOpTree::
LIsNull() const
    {
    if(L_ == nullptr) return true;
    return !bool(*L_);
    }

bool inline LocalOpTree::
RIsNull() const
    {
    if(R_ == nullptr) return true;
    return !bool(*R_);
    }

void inline LocalOpTree::
product(ITensor const& phi, 
        ITensor      & phip) const
    {
    if(!(*this)) Error("LocalOpTree is null");

    if(LIsNull())
        {
        phip = phi;
        if(!RIsNull()) 
            phip *= R(); //m^3 k d
        
        if(nc_ == 2)
            {
            phip *= (*Op2_); //m^2 k^2
            phip *= (*Op1_); //m^2 k^2
            }
        else if(nc_ == 1)
            {
            phip *= (*Op1_);
            }
        }
    else
        {
        phip = phi * L(); //m^3 k d

        if(nc_ == 2)
            {
            phip *= (*Op1_); //m^2 k^2
            phip *= (*Op2_); //m^2 k^2
            }
        else if(nc_ == 1)
            {
            phip *= (*Op1_);
            }

        if(!RIsNull()) 
            phip *= R();
        }

    phip.noPrime();
    }

Real inline LocalOpTree::
expect(const ITensor& phi) const
    {
    ITensor phip;
    product(phi,phip);
    return real(eltC(dag(phip) * phi));
    }

ITensor inline LocalOpTree:: // TODO To be updated to work with 0 or 1 center site and 
deltaRho(ITensor const& AA, 
         ITensor const& combine, 
         Direction dir) const
    {
    if(nc_ != 2)
        {
        Error("LocalMPO: currently only support 2 center sites in deltaRho");
        }

    auto drho = AA;
    if(dir == Fromleft)
        {
        if(!LIsNull()) drho *= L();
        drho *= (*Op1_);
        }
    else //dir == Fromright
        {
        if(!RIsNull()) drho *= R();
        drho *= (*Op2_);
        }
    drho.noPrime();
    drho = combine * drho;
    auto ci = commonIndex(combine,drho);
    drho *= dag(prime(drho,ci));

    //Expedient to ensure drho is Hermitian
    drho = drho + dag(swapTags(drho,"0","1"));
    drho /= 2.;

    return drho;
    }

ITensor inline LocalOpTree::
diag() const 
    {
    if(!(*this)) Error("LocalOpTree is default constructed");

    //lambda helper function:
    auto findIndPair = [](ITensor const& T) {
        for(auto& s : T.inds())
            {
            if(s.primeLevel() == 0 && hasIndex(T,prime(s))) 
                {
                return s;
                }
            }
        return Index();//default constructed
        };

    Index toTie;
    ITensor Diag;
    if(nc_ == 2)
        {
        auto& Op1 = *Op1_;
        auto& Op2 = *Op2_;

        toTie = findIndPair(Op1);
        if(toTie)
            {
            Diag = Op1 * delta(toTie,prime(toTie),prime(toTie,2));
        	Diag.noPrime();
            }
        else
            {
            Diag = Op1;
            }

        toTie = findIndPair(Op2);
		if(toTie)
            {
            auto Diag2 = Op2 * delta(toTie,prime(toTie),prime(toTie,2));
        	Diag *= noPrime(Diag2);
            }
        else
            {
            Diag *= Op2;
            }
        }
    else if(nc_ == 1)
        {
        auto& Op1 = *Op1_;
		toTie = findIndPair(Op1);
        if(toTie)
            {
            Diag = Op1 * delta(toTie,prime(toTie),prime(toTie,2));
        	Diag.noPrime();
            }
        else
            {
            Diag = Op1;
            }
        
        }

    if(!LIsNull())
        {
        toTie = findIndPair(L());
        if(toTie)
            {
            auto DiagL = L() * delta(toTie,prime(toTie),prime(toTie,2));
            if(Diag) Diag *= noPrime(DiagL);
            else Diag = noPrime(DiagL);
            }
        else
            {
            if(Diag) Diag *= L();
            else Diag = L();
            }
        }

    if(!RIsNull())
        {
        toTie = findIndPair(R());
        if(toTie)
            {
            auto DiagR = R() * delta(toTie,prime(toTie),prime(toTie,2));
            if(Diag) Diag *= noPrime(DiagR);
            else Diag = noPrime(DiagR);
            }
        else
            {
            if(Diag) Diag *= R();
			else Diag = R();
            }
        }

    Diag.dag();
    //Diag must be real since operator assumed Hermitian
    Diag.takeReal();

    return Diag;
    }

size_t inline LocalOpTree::
size() const
    {
    if(!(*this)) Error("LocalOpTree is default constructed");
    if(size_ == size_t(-1))
        {
        //Calculate linear size of this 
        //op as a square matrix
        size_ = 1;
        if(!LIsNull()) 
            {
            for(auto& I : L().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= dim(I);
                    break;
                    }
                }
            }
        if(!RIsNull()) 
            {
            for(auto& I : R().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= dim(I);
                    break;
                    }
                }
            }
        if(nc_ == 2)
            {
				for(auto& I : Op1().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= dim(I);
                    break;
                    }
                }
				for(auto& I : Op2().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= dim(I);
                    break;
                    }
                }
            }
        else if(nc_ == 1)
            {
            for(auto& I : Op1().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= dim(I);
                    break;
                    }
                }
            }
        }
    return size_;
    }

} //namespace itensor

#endif
