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
#ifndef __ITENSOR_BINARYTREE_IMPL_H_
#define __ITENSOR_BINARYTREE_IMPL_H_

namespace itensor {

  template <typename BigMatrixT>
  Spectrum BinaryTree::
  svdBond(int b1, ITensor const& AA,int b2,
	  BigMatrixT const& PH, Args args)
  {
    //Check for out of range nodes
    if(( b1 <0 || b2 <0) || (b1 > N_ || b2 > N_))
      {
	printfln("svdBound: b1=%d, b2=%d",b1,b2);
        Error("b1 or b2 out of range");
      }
    //Check if is a common Index
    if (!commonIndex(A_[b1],A_[b2]))
      {
	printfln("svdBond b1=%d, b2=%d",b1,b2);
        Error("b1 and b2 don't have a common Index");
      }

    auto noise = args.getReal("Noise",0.);
    auto cutoff = args.getReal("Cutoff",MIN_CUT);
    auto usesvd = args.getBool("UseSVD",false);
    // Truncate blocks of degenerate singular values
    args.add("RespectDegenerate",args.getBool("RespectDegenerate",true));

    Spectrum res;

    // Store the original tags for link b1-> b2 so that it can
    // be put back onto the newly introduced link index
    auto original_link_tags = tags(commonIndex(A_[b1],A_[b2]));

    if(usesvd || (noise == 0 && cutoff < 1E-12))
      {
        //Need high accuracy, use svd which calls the
        //accurate SVD method in the MatrixRef library
        ITensor D;
        res = svd(AA,A_[b1],D,A_[b2],args);

        // int dim = std::min((int)std::pow(site_dim_, pow2(height() - depth(b2))), (int)args.getInt("MaxDim", MAX_DIM));
        // if (commonIndex(A_[b1], D).dim() < dim) {
        //   auto temp1 = A_[b1];
        //   auto temp2 = D;
        //   auto ind = Index(dim, "Link,U"); //TODO: Create an Index with QNs
        //   A_[b1] = ITensor(IndexSet(uniqueInds(temp1, temp2), ind));
        //   D = ITensor(IndexSet(uniqueInds(temp2, temp1), ind));
        //   for(auto it : iterInds(temp1)) {
        //     A_[b1].set(it[0].val, it[1].val, it[2].val, temp1.real(it));
        //   }
        //   for(auto it : iterInds(temp2)) {
        //     D.set(it[1].val, it[0].val, temp2.real(it));
        //   }
				//
        //   auto bnd = commonIndex(A_[b1], D);
        //   ITensor A, B(bnd);
        //   qr(A_[b1], A, B);
        //   A_[b1] = A;
        //   D *= B;
        //   auto bnd_qr = commonIndex(A_[b1], D);
        //   A_[b1].replaceInds({bnd_qr}, {bnd});
        //   D.replaceInds({bnd_qr}, {bnd});
        // }

        //Normalize the ortho center if requested
        if(args.getBool("DoNormalize",false))
	  {
            D *= 1./itensor::norm(D);
	  }

        //Push the singular values into the appropriate site tensor
        A_[b2] *= D;
      }
    else
      {
        //If we don't need extreme accuracy
        //or need to use noise term
        //use density matrix approach
        res = denmatDecomp(AA,A_[b1],A_[b2],Fromleft,PH,args);
        //Normalize the ortho center if requested
        if(args.getBool("DoNormalize",false))
	  {
            auto nrm = itensor::norm(A_[b2]);
            if(nrm > 1E-16) A_[b2] *= 1./nrm;
	  }
      }

    // Put the old tags back onto the new index
    auto lb = commonIndex(A_[b1],A_[b2]);
    A_[b1].setTags(original_link_tags,lb);
    A_[b2].setTags(original_link_tags,lb);

    //Update orth_pos_
    orth_pos_.at(b1) = b2;// We update the orthogonalisation memory
    orth_pos_.at(b2) = -1; // The next one is not any more orthogonal

    return res;
  }

} //namespace itensor

#endif
