#include "base.h"
#include "matrix.h"
#include "model.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using mnistnn::Pixel;
using mnistnn::Vector;
using mnistnn::Model;
using mnistnn::ForwardCache;
using mnistnn::linalg::Matrix;
using mnistnn::linalg::fill;

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

static void testModelForward(float tol)
{
    Model model(2);

    fill(model.W0, 0.0f);
    fill(model.b0, 0.0f);
    fill(model.W1, 0.0f);
    fill(model.b1, 0.0f);

    model.W0.data[0] = 1.0f;                    // hidden 0 reads pixel 0
    model.W0.data[model.num_input + 1] = 1.0f;  // hidden 1 reads pixel 1

    model.W1.data[0] = 2.0f;  // output 0 uses hidden 0
    model.W1.data[1] = 3.0f;  // output 0 uses hidden 1

    model.W1.data[model.num_hidden] = -1.0f;     // output 1 uses hidden 0
    model.W1.data[model.num_hidden + 1] = 4.0f;  // output 1 uses hidden 1

    Vector<Pixel> input(model.num_input, 0);
    input[0] = 255; // normalized to 1.0
    input[1] = 0;   // normalized to 0.0

    Matrix logits = model.forward(input);

    const float h0 = 1.0f / (1.0f + std::exp(-1.0f)); // sigmoid(1)
    const float h1 = 0.5f;                            // sigmoid(0)

    requireTrue(logits.rows == model.num_output, "forward output rows");
    requireTrue(logits.cols == 1, "forward output cols");
    requireNear(logits.data[0], 2.0f * h0 + 3.0f * h1, tol, "logit 0");
    requireNear(logits.data[1], -1.0f * h0 + 4.0f * h1, tol, "logit 1");
}

static void testModelForwardIncorrectInputShape()
{
    Model model(2);

    Vector<Pixel> input(model.num_input -1, 0);

    bool exception_thrown{ false };

    try
    {
        model.forward(input);
    }
    catch(const std::invalid_argument&)
    {
        exception_thrown = true;
    }
    
    requireTrue(exception_thrown, "forward should reject incorrect input shape.");
}


int main()
{
    testModelForward(1e-6f);
    testModelForwardIncorrectInputShape();

    std::cout << "All model tests passed\n";

    return 0;
}