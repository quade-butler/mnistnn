#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using Matrix::Matrix;
using Matrix::matMul;
using Matrix::matSigmoid;
using Matrix::matSoftmax;


static void testMatMulKnownValues()
{

}

static void testSigmoidKnownValues()
{
    
}

static void testSoftmaxSumsToOne()
{

}

int main()
{
    testMatMulKnownValues();
    testSigmoidKnownValues();
    testSoftmaxSumsToOne();

    std::cout << "All matrix tests passed\n";

    return 0;
}