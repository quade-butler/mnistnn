#include "train.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "base.h"
#include "matrix.h"

using mnistnn::linalg::Matrix;
using mnistnn::linalg::fill;
using mnistnn::Label;
using mnistnn::Size;
using mnistnn::crossEntropyLoss;
using mnistnn::crossEntropyGradLogits;

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
        crossEntropyLoss(logits, target);
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
    logits.data = { 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    float loss{ crossEntropyLoss(logits, target)};
    float expected_loss{ std::log(std::exp(2.0f) + 9.0f * std::exp(0.0f)) - 2.0f };

    requireNear(loss, expected_loss, tol, "crossEntropyLoss known logits");
}

static void testCrossEntropyLossLargeLogits(float tol)
{
    Label target{ 1 };

    Matrix logits(10, 1);
    logits.data = { 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f, 1e3f };

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

static void testCrossEntropyGradEqualLogits(float tol)
{
    Matrix logits(10, 1);
    fill(logits, 0.0f);

    const Label target{3};
    const Matrix gradient{crossEntropyGradLogits(logits, target)};

    requireTrue(
        gradient.rows == 10 && gradient.cols == 1,
        "cross-entropy gradient should preserve logits shape"
    );

    float sum{0.0f};

    for (Size i{0}; i < gradient.data.size(); ++i)
    {
        const float expected{i == target ? -0.9f : 0.1f};
        requireNear(gradient.data[i],expected,tol,"cross-entropy gradient value");
        sum += gradient.data[i];
    }

    requireNear(sum, 0.0f, tol, "cross-entropy gradient should sum to zero");
}

static void testCrossEntropyGradientNumerically(float tol)
{
    Matrix logits(10, 1);
    logits.data = { -1.0f, 0.5f, 2.0f, -0.5f, 1.0f, 0.0f, 0.25f, -2.0f, 1.5f, 0.75f };

    const Label target{2};
    const Matrix analytical{crossEntropyGradLogits(logits, target)};

    constexpr float epsilon{1e-3f};

    for (Size i{0}; i < logits.data.size(); ++i)
    {
        Matrix plus{logits};
        Matrix minus{logits};

        plus.data[i] += epsilon;
        minus.data[i] -= epsilon;

        const float numerical{ 
            (crossEntropyLoss(plus, target) - crossEntropyLoss(minus,target)) 
            / (2.0f * epsilon)
        };

        requireNear(
            analytical.data[i], 
            numerical, 
            tol, 
            "cross-entropy analytical and numerical gradients"
        );
    }


}

int main()
{
    testCrossEntropyLoss(1e-6f);
    testCrossEntropyInvalidLogitsShape();
    testCrossEntropyTargetOutOfBounds();
    testCrossEntropyEmptyLogitsData();
    testCrossEntropyLossLargeLogits(1e-6);
    testCrossEntropyGradEqualLogits(1e-6f);
    testCrossEntropyGradientNumerically(1e-4f);

    std::cout << "All training tests passed\n";

    return 0;
}