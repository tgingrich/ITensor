//
// Distributed under the ITensor Library License, Version 1.2
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_INDEXSET_IMPL_H
#define __ITENSOR_INDEXSET_IMPL_H

namespace itensor {

template<typename... VarArgs>
void IndexSet::
swapTags(TagSet const& ts1, 
         TagSet const& ts2, 
         VarArgs&&... vargs)
    {
    TagSet tempTags;
    if(primeLevel(ts1) < 0)
      {
      if(primeLevel(ts2) >= 0) Error("Error in swapTags: cannot swap integer tag with non-integer tag");
      tempTags = TagSet("df4sd32");
      }
    else
      {
      if(primeLevel(ts2) < 0) Error("Error in swapTags: cannot swap integer tag with non-integer tag");
      tempTags = TagSet("df4sd32,431543");
      }
#ifdef DEBUG
    for(auto& I : *this)
        {
        if(hasTags(I,tempTags))
            {
            println("tempTags = ",tempTags);
            throw ITError("swapTags fails if an index has tags tempTags");
            }
        }
#endif
    this->replaceTags(ts1,tempTags,std::forward<VarArgs>(vargs)...);
    this->replaceTags(ts2,ts1,std::forward<VarArgs>(vargs)...);
    this->replaceTags(tempTags,ts2);
    }

template<class LabelT>
void
contractIS(IndexSet const& Lis,
           LabelT const& Lind,
           IndexSet const& Ris,
           LabelT const& Rind,
           IndexSet & Nis,
           LabelT & Nind,
           bool sortResult)
    {
    //Don't allow mixing of Indices with and without QNs
    auto LhasQNs = hasQNs(Lis);
    auto RhasQNs = hasQNs(Ris);
    auto LremoveQNs = (LhasQNs && !RhasQNs);
    auto RremoveQNs = (!LhasQNs && RhasQNs);

    long ncont = 0;
    for(auto& i : Lind) if(i < 0) ++ncont;
    auto nuniq = Lis.order()+Ris.order()-2*ncont;
    auto newind = IndexSetBuilder(nuniq);
    //Below we are "cheating" and using the .str
    //field of each member of newind to hold the 
    //labels which will go into Nind so they will
    //be sorted along with the .ext members (the indices of Nis)
    //Later we will set them back to zero
    //IndexSetT constructor anyway
    for(decltype(Lis.order()) j = 0; j < Lis.order(); ++j)
        {
        if(Lind[j] > 0) //uncontracted
            {
            if(LremoveQNs) newind.nextIndStr(removeQNs(Lis[j]),Lind[j]);
            else newind.nextIndStr(Lis[j],Lind[j]);
            }
        }
    for(decltype(Ris.order()) j = 0; j < Ris.order(); ++j)
        {
        if(Rind[j] > 0) //uncontracted
            {
            if(RremoveQNs) newind.nextIndStr(removeQNs(Ris[j]),Rind[j]);
            else newind.nextIndStr(Ris[j],Rind[j]);
            }
        }
    if(sortResult) newind.sortByIndex();
    Nind.resize(newind.size());
    for(decltype(newind.size()) j = 0; j < newind.size(); ++j)
        {
        Nind[j] = newind.stride(j);
        }
    Nis = newind.build();
    Nis.computeStrides();
    }

template<class LabelT>
void
contractISReplaceIndex(IndexSet const& Lis,
                       LabelT const& Lind,
                       IndexSet const& Ris,
                       LabelT const& Rind,
                       IndexSet & Nis)
    {
    auto newind = IndexSetBuilder(Lis.order());
    for( auto i : range(Lis.order()) )
        {
        if( Lind[i] < 0 )
            {
            if( Rind[0] >= 0 ) newind.nextIndex(Ris[0]);
            else newind.nextIndex(Ris[1]);
            }
        else
            {
            newind.nextIndex(Lis[i]);
            }
        }
    Nis = newind.build();
    }

template<class LabelT>
void
ncprod(IndexSet const& Lis,
       LabelT const& Lind,
       IndexSet const& Ris,
       LabelT const& Rind,
       IndexSet & Nis,
       LabelT & Nind)
    {
    long nmerge = 0;
    for(auto& i : Lind) if(i < 0) ++nmerge;
    auto nuniq = Lis.order()+Ris.order()-2*nmerge;
    auto NisBuild = RangeBuilderT<IndexSet>(nmerge+nuniq);
    Nind.resize(nmerge+nuniq);
    long n = 0;

    for(decltype(Lis.order()) j = 0; j < Lis.order(); ++j)
        {
        if(Lind[j] < 0) //merged
            {
            NisBuild.nextIndex(Lis[j]);
            Nind[n++] = Lind[j];
            }
        }
    for(decltype(Lis.order()) j = 0; j < Lis.order(); ++j)
        {
        if(Lind[j] > 0) //unmerged/unique
            {
            NisBuild.nextIndex(Lis[j]);
            Nind[n++] = Lind[j];
            }
        }
    for(decltype(Ris.order()) j = 0; j < Ris.order(); ++j)
        {
        if(Rind[j] > 0) //unmerged/unique
            {
            NisBuild.nextIndex(Ris[j]);
            Nind[n++] = Rind[j];
            }
        }
    Nis = NisBuild.build();
    }

} //namespace itensor

#endif
