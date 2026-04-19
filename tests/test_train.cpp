#include "base.h"
#include "matrix.h"
#include "train.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using mnistnn::linalg::Matrix;
using mnistnn::Label;
using mnistnn::crossEntropyLoss;

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

void testCrossEntropyLoss(float tol)
{
    Label target{ 1 };

    Matrix logits(10, 1);
    logits.data = {
        0.0f,
        2.0f,  // target class
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f
    };

    float loss{ crossEntropyLoss(logits, target)};
    float expected_loss{ std::log(std::exp(2) + 9 * std::exp(0)) - 2 };
    requireNear(loss, expected_loss, tol, "crossEntropyLoss known logits");
}

int main()
{
    testCrossEntropyLoss(1e-6f);

    std::cout << "All training tests passed\n";

    return 0;
}