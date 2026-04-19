#include "matrix.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using mnistnn::linalg::Matrix;

static void requireNear(
    float actual, float expected, float tol, const char* msg
)
{
    if (std::fabs(actual - expected) > tol)
    {
        std::cerr << msg << ": expected " << expected 
                  << ", got " << actual << '\n';
        throw std::runtime_error("test failed");
    }
}

static void requireTrue(bool condition, const char* msg)
{
    if (!condition)
    {
        std::cerr << msg << '\n';
        throw std::runtime_error("test failed");
    }
}

static void testMatMulKnownValues(float tol)
{
    Matrix a(2, 3);
    a.data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Matrix b(3, 2);
    b.data = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    Matrix out(2, 2);
    matMul(out, a, b);

    requireNear(out.data[0], 58.0f,  tol, "matMul[0,0]");
    requireNear(out.data[1], 64.0f,  tol, "matMul[0,1]");
    requireNear(out.data[2], 139.0f, tol, "matMul[1,0]");
    requireNear(out.data[3], 154.0f, tol, "matMul[1,1]");
}

static void testMatMulTransposeNT(float tol)
{
    Matrix a(2, 3);
    a.data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Matrix b(4, 3);
    b.data = {
        7.0f, 11.0f, 15.0f,
        8.0f, 12.0f, 16.0f,
        9.0f, 13.0f, 17.0f,
        10.0f, 14.0f, 18.0f,
    };
    
    Matrix out(2, 4);
    matMul(out, a, b, false, true);

    requireNear(out.data[0], 74.0f,  tol, "matMul[0,0]");
    requireNear(out.data[1], 80.0f,  tol, "matMul[0,1]");
    requireNear(out.data[2], 86.0f,  tol, "matMul[0,2]");
    requireNear(out.data[3], 92.0f,  tol, "matMul[0,3]");
    requireNear(out.data[4], 173.0f, tol, "matMul[1,0]");
    requireNear(out.data[5], 188.0f, tol, "matMul[1,1]");
    requireNear(out.data[6], 203.0f, tol, "matMul[1,2]");
    requireNear(out.data[7], 218.0f, tol, "matMul[1,3]");
}

static void testMatMulTransposeTN(float tol)
{
    Matrix a(3, 2);
    a.data = {
        1.0f, 4.0f,  
        2.0f, 5.0f,  
        3.0f, 6.0f
    };

    Matrix b(3, 4);
    b.data = {
        7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f, 18.0f,
    };

    Matrix out(2, 4);
    matMul(out, a, b, true, false);

    requireNear(out.data[0], 74.0f,  tol, "matMul[0,0]");
    requireNear(out.data[1], 80.0f,  tol, "matMul[0,1]");
    requireNear(out.data[2], 86.0f,  tol, "matMul[0,2]");
    requireNear(out.data[3], 92.0f,  tol, "matMul[0,3]");
    requireNear(out.data[4], 173.0f, tol, "matMul[1,0]");
    requireNear(out.data[5], 188.0f, tol, "matMul[1,1]");
    requireNear(out.data[6], 203.0f, tol, "matMul[1,2]");
    requireNear(out.data[7], 218.0f, tol, "matMul[1,3]");
}

static void testMatMulTransposeTT(float tol)
{
    Matrix a(3, 2);
    a.data = {
        1.0f, 4.0f,  
        2.0f, 5.0f,  
        3.0f, 6.0f
    };

    Matrix b(4, 3);
    b.data = {
        7.0f, 11.0f, 15.0f,
        8.0f, 12.0f, 16.0f,
        9.0f, 13.0f, 17.0f,
        10.0f, 14.0f, 18.0f,
    };

    Matrix out(2, 4);
    matMul(out, a, b, true, true);

    requireNear(out.data[0], 74.0f,  tol, "matMul[0,0]");
    requireNear(out.data[1], 80.0f,  tol, "matMul[0,1]");
    requireNear(out.data[2], 86.0f,  tol, "matMul[0,2]");
    requireNear(out.data[3], 92.0f,  tol, "matMul[0,3]");
    requireNear(out.data[4], 173.0f, tol, "matMul[1,0]");
    requireNear(out.data[5], 188.0f, tol, "matMul[1,1]");
    requireNear(out.data[6], 203.0f, tol, "matMul[1,2]");
    requireNear(out.data[7], 218.0f, tol, "matMul[1,3]");
}

static void testMatMulIncorrectInputShape()
{
    Matrix a(2, 3);
    a.data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Matrix b(2, 3);
    b.data = {
        7.0f, 9.0f, 11.0f,
        8.0f, 10.0f, 12.0f,
    };

    Matrix out(2, 2);

    bool exception_thrown { false };
    try { 
        matMul(out, a, b); 
    } 
    catch (const std::invalid_argument& e)
    {
        exception_thrown = true;
    }

    if (!exception_thrown)
    {
        std::cerr << "Input shapes were indeed correct.";
        throw std::runtime_error("test failed");
    }
}

static void testMatMulIncorrectOutputShape()
{
    Matrix a(2, 3);
    a.data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Matrix b(3, 2);
    b.data = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };

    Matrix out(2, 3);  // should be (2, 2)

    bool exception_thrown { false };
    try { 
        matMul(out, a, b); 
    } 
    catch (const std::invalid_argument& e)
    {
        exception_thrown = true;
    }

    if (!exception_thrown)
    {
        std::cerr << "Output shape was indeed correct.";
        throw std::runtime_error("test failed");
    }
}

static void testSigmoidKnownValues(float tol)
{
    Matrix x(3,1);
    x.data = { 0.0f, 2.0f, -2.0f };

    Matrix y{ sigmoid(x) };

    requireNear(y.data[0], 0.5f, tol, "sigmoid(0)");
    requireTrue(y.data[1] > 0.5f, "sigmoid(positive) should be > 0.5");
    requireTrue(y.data[2] < 0.5f, "sigmoid(negative) should be > 0.5");
}

static void testSoftmaxSumsToOne(float tol)
{
    Matrix x(3,1);
    x.data = { 1.0f, 2.0f, 3.0f };
    
    Matrix y{ softmax(x) };

    float sum { y.data[0] + y.data[1] + y.data[2]};

    requireNear(sum, 1.0f, tol, "softmax sum");
    requireTrue(y.data[2] > y.data[1], "larger logit should have larger probability");
    requireTrue(y.data[1] > y.data[0], "larger logit should have larger probability");
}

static void testReluKnownValues(float tol)
{
    Matrix x(2, 3);
    x.data = {
        -2.0f, -0.5f, 0.0f,
         1.0f, 3.5f, 8.0f
    };

    Matrix y{ relu(x) };

    requireTrue(y.rows == 2 && y.cols == 3, "relu should preserve shape");
    requireNear(y.data[0], 0.0f, tol, "relu(-2)");
    requireNear(y.data[1], 0.0f, tol, "relu(-0.5)");
    requireNear(y.data[2], 0.0f, tol, "relu(0)");
    requireNear(y.data[3], 1.0f, tol, "relu(1)");
    requireNear(y.data[4], 3.5f, tol, "relu(3.5)");
    requireNear(y.data[5], 8.0f, tol, "relu(8)");
}

int main()
{
    testMatMulKnownValues(1e-6f);
    testMatMulTransposeNT(1e-6f);
    testMatMulTransposeTN(1e-6f);
    testMatMulTransposeTT(1e-6f);
    testMatMulIncorrectInputShape();
    testMatMulIncorrectOutputShape();
    testSigmoidKnownValues(1e-6f);
    testSoftmaxSumsToOne(1e-6f);
    testReluKnownValues(1e-6f);

    std::cout << "All matrix tests passed\n";

    return 0;
}