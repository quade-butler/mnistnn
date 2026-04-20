#include "base.h"
#include "matrix.h"
#include "train.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

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

static void requireThrowsInvalidArgument(Matrix logits, Label target, const char* msg)
{
    bool exception_thrown{ false };
    try
    {
        crossEntropyLoss(logits, target);  // change to pointer or use std::function
    }
    catch(const std::invalid_argument&)
    {
        exception_thrown = true;
    }

    requireTrue(exception_thrown, msg);
}

static void testCrossEntropyLoss(float tol)
{
    Label target{ 1 };

    Matrix logits(10, 1);
    logits.data = {
        0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    float loss{ crossEntropyLoss(logits, target)};
    float expected_loss{ std::log(std::exp(2.0f) + 9.0f * std::exp(0.0f)) - 2.0f };

    requireNear(loss, expected_loss, tol, "crossEntropyLoss known logits");
}

static void testCrossEntropyLossLargeLogits(float tol)
{
    Label target{ 1 };

    Matrix logits(10, 1);
    logits.data = { 
        1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f 
    };

    float loss{ crossEntropyLoss(logits, target)};
    float expected_loss{ std::log(10.0f) };

    requireTrue(std::isfinite(loss), "crossEntropyLoss should be finite for large logits");
    requireNear(loss, expected_loss, tol, "crossEntropyLoss large equal logits");
}

static void testCrossEntropyInvalidLogitsShape()
{
    Label target{ 1 };
    Matrix logits(1, 10);

    const char* msg = "crossEntropyLoss should reject input shape";
    requireThrowsInvalidArgument(logits, target, msg);
}

static void testCrossEntropyTargetOutOfBounds()
{
    Label target{ 10 };
    Matrix logits(10, 1);  // 

    const char* msg = "crossEntropyLoss should reject target out of bounds";
    requireThrowsInvalidArgument(logits, target, msg);
}

static void testCrossEntropyEmptyLogitsData()
{
    Label target{ 1 };
    Matrix logits(10, 1);
    logits.data = {};

    const char* msg = "crossEntropyLoss should reject empty logits";
    requireThrowsInvalidArgument(logits, target, msg);
}

int main()
{
    testCrossEntropyLoss(1e-6f);
    testCrossEntropyInvalidLogitsShape();
    testCrossEntropyTargetOutOfBounds();
    testCrossEntropyEmptyLogitsData();

    std::cout << "All training tests passed\n";

    return 0;
}