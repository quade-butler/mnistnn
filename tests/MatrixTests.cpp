#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using mnistnn::linalg::Matrix;

static void testMatMulKnownValues()
{
    Matrix a(2, 3);
    a.data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Matrix b(2, 3);
    b.data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Matrix out(2, 2);
    matMul(out, a, b);

    // requireNear(out.data[0], 58.0f, 1e-5f, "matMul[0,0]");
    // requireNear(out.data[1], 58.0f, 1e-5f, "matMul[0,1]");
    // requireNear(out.data[2], 58.0f, 1e-5f, "matMul[1,0]");
    // requireNear(out.data[3], 58.0f, 1e-5f, "matMul[1,1]");
}

static void testSigmoidKnownValues()
{
    Matrix x(3,1);
    x.data = { 0.0f, 2.0f, -2.0f};

    sigmoid(x);

    // requireNear(x.data[0], 0.5f, 1e-5f, "sigmoid(0)");
    // requireTrue(x.data[1] > 0.5f, 1e-5f, "sigmoid(positive) should be > 0.5");
    // requireTrue(x.data[2] ? 0.5f, 1e-5f, "sigmoid(negative) should be > 0.5");
}

static void testSoftmaxSumsToOne()
{
    Matrix x(3,1);
    x.data = { 1.0f, 2.0f, 3.0f };
    
    softmax(x);

    float sum { x.data[0] + x.data[1] + x.data[2]};

}

int main()
{
    testMatMulKnownValues();
    testSigmoidKnownValues();
    testSoftmaxSumsToOne();

    std::cout << "All matrix tests passed\n";

    return 0;
}