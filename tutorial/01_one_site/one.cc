//
// 2019 Many Electron Collaboration Summer School
// ITensor Tutorial
//
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

using namespace itensor;
using namespace Spectra;

class MyDiagonalTen
{
public:
    int rows() { return 10; }
    int cols() { return 10; }
    void perform_op(const double *x_in, double *y_out)
    {
        for(int i = 0; i < rows(); i++)
        {
            y_out[i] = x_in[i] * (i + 1);
        }
    }
};

int main()
    {
Real real = 0.2;
double d = real;
std::cout << d << std::endl;
MyDiagonalTen op;
    SymEigsSolver<double, LARGEST_ALGE, MyDiagonalTen> eigs(&op, 3, 6);
    eigs.init();
    eigs.compute();
    if(eigs.info() == SUCCESSFUL)
    {
        Eigen::VectorXd evalues = eigs.eigenvalues();
        std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    }
    //
    // Define our Index 
    // (the space we are working with)
    //
auto i = Index(3);
auto j = Index(5);
auto k = Index(7);
auto l = Index(9);

auto T = randomITensor(i,j,k,l);

auto [C,c] = combiner(i,k);

//
// Combine
// 
auto cT = C * T; //or T * C, which has same effect
PrintData(T);
PrintData(cT);
    auto s = Index(2,"s");

    //
    // Operators 
    //

    auto Sx = ITensor(s,prime(s));

    Sx.set(s=1,prime(s)=2,+0.5);
    Sx.set(s=2,prime(s)=1,+0.5);

    PrintData(Sx);

    //
    // Single-site wavefunction
    //
    
    auto psi = ITensor(s); //initialized to zero

    //
    // TODO 
    //
    // 1. make the above wavefunction
    //    the (normalized) positive Sx eigenstate
    //    HINT: use psi.set(...)
    //

    /* Your code here */

    PrintData(psi);
    
    //
    // TODO
    //
    // 2. Compute |phi> = Sx |psi> using
    //    the Sx and psi ITensors above
    //    AND
    //    compute: auto olap = <psi|phi>
    //    using the * operator and elt(...) method.
    //    Print the result with PrintData(...).
    //

    /* Your code here */

    //
    // TODO
    //
    // 3. Try normalizing |phi> and recompute
    //    the inner product <psi|phi>
    //    Print the result with PrintData(...).
    //    HINT: use phi /= norm(phi)) to normalize.
    //

    /* Your code here */

    return 0;
    }
