#include "model.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "base.h"
#include "matrix.h"
#include "train.h"

using mnistnn::Pixel;
using mnistnn::Label;
using mnistnn::Size;
using mnistnn::Vector;
using mnistnn::Model;
using mnistnn::ForwardCache;
using mnistnn::linalg::Matrix;
using mnistnn::linalg::fill;
using mnistnn::ModelGradients;
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

static void testModelBackward(float tol)
{
    Model model(2);

    fill(model.W1, 0.0f);

    // W1:
    // [1 2]
    // [3 4]
    // [0 0]
    // ...
    model.W1.data[0] = 1.0f;
    model.W1.data[1] = 2.0f;
    model.W1.data[2] = 3.0f;
    model.W1.data[3] = 4.0f;

    ForwardCache cache{
        Matrix(model.num_input, 1),
        Matrix(model.num_hidden, 1),
        Matrix(model.num_hidden, 1),
        Matrix(model.num_output, 1),
    };

    cache.x.data[0] = 1.0f;
    cache.x.data[1] = 0.5f;

    cache.a1.data = {0.25f, 0.75f};

    Matrix dlogits(model.num_output, 1);
    dlogits.data[0] = 0.2f;
    dlogits.data[1] = -0.2f;

    const ModelGradients gradients{ model.backward(cache, dlogits) };

    requireTrue(
        gradients.dW1.rows == model.num_output &&
        gradients.dW1.cols == model.num_hidden,
        "dW1 shape"
    );
    
    requireTrue(
        gradients.db1.rows == model.num_output &&
        gradients.db1.cols == 1,
        "db1 shape"
    );
    
    requireTrue(
        gradients.dW0.rows == model.num_hidden &&
        gradients.dW0.cols == model.num_input,
        "dW0 shape"
    );
    
    requireTrue(
        gradients.db0.rows == model.num_hidden &&
        gradients.db0.cols == 1,
        "db0 shape"
    );

    requireNear(gradients.dW1.data[0], 0.05f, tol, "dW1[0,0]");
    requireNear(gradients.dW1.data[1], 0.15f, tol, "dW1[0,1]");
    requireNear(gradients.dW1.data[2], -0.05f, tol, "dW1[1,0]");
    requireNear(gradients.dW1.data[3], -0.15f, tol, "dW1[1,1]");

    requireNear(gradients.db1.data[0], 0.2f, tol, "db1[0]");
    requireNear(gradients.db1.data[1], -0.2f, tol, "db1[1]");

    requireNear(gradients.db0.data[0], -0.075f, tol, "db0[0]");
    requireNear(gradients.db0.data[1], -0.075f, tol, "db0[1]");

    requireNear(gradients.dW0.data[0], -0.075f, tol, "dW0[0,0]");
    requireNear(gradients.dW0.data[1], -0.0375f, tol, "dW0[0,1]");

    const Size second_row(model.num_input);

    requireNear(
        gradients.dW0.data[second_row],
        -0.075f,
        tol,
        "dW0[1,0]"
    );
    requireNear(
        gradients.dW0.data[second_row + 1],
        -0.0375f,
        tol,
        "dW0[1,1]"
    );
}

static void testModelBackwardNumerically(float tol)
{
    Model model(2);

    fill(model.W0, 0.0f);
    fill(model.b0, 0.0f);
    fill(model.W1, 0.0f);
    fill(model.b1, 0.0f);

    // Give both hidden nodes small, deterministic pre-activations.
    model.W0.data[0] = 0.2f;
    model.W0.data[1] = -0.1f;

    const Size second_hidden_row{model.num_input};
    model.W0.data[second_hidden_row] = -0.3f;
    model.W0.data[second_hidden_row + 1] = 0.4f;

    model.b0.data[0] = 0.05f;
    model.b0.data[1] = -0.02f;

    // First two output neurons depend on both hidden neurons.
    model.W1.data[0] = 0.3f;
    model.W1.data[1] = -0.2f;
    model.W1.data[2] = -0.1f;
    model.W1.data[3] = 0.4f;

    Vector<Pixel> input(model.num_input, 0);
    input[0] = 255;
    input[1] = 128;

    const Label target{1};

    const ForwardCache cache{model.forwardWithCache(input)};
    const Matrix dlogits{
        crossEntropyGradLogits(cache.logits, target)
    };
    const ModelGradients analytical{
        model.backward(cache, dlogits)
    };

    constexpr float epsilon{1e-3f};

    const auto numericalGradient =
        [&](float& parameter)
        {
            const float original{parameter};

            parameter = original + epsilon;
            const float loss_plus{
                crossEntropyLoss(model.forward(input), target)
            };

            parameter = original - epsilon;
            const float loss_minus{
                crossEntropyLoss(model.forward(input), target)
            };

            parameter = original;

            return (loss_plus - loss_minus) / (2.0f * epsilon);
        };

    requireNear(
        analytical.dW0.data[0],
        numericalGradient(model.W0.data[0]),
        tol,
        "numerical dW0[0,0]"
    );

    requireNear(
        analytical.dW0.data[second_hidden_row],
        numericalGradient(model.W0.data[second_hidden_row]),
        tol,
        "numerical dW0[1,0]"
    );

    requireNear(
        analytical.db0.data[0],
        numericalGradient(model.b0.data[0]),
        tol,
        "numerical db0[0]"
    );

    requireNear(
        analytical.dW1.data[0],
        numericalGradient(model.W1.data[0]),
        tol,
        "numerical dW1[0,0]"
    );

    requireNear(
        analytical.dW1.data[2],
        numericalGradient(model.W1.data[2]),
        tol,
        "numerical dW1[1,0]"
    );

    requireNear(
        analytical.db1.data[1],
        numericalGradient(model.b1.data[1]),
        tol,
        "numerical db1[1]"
    );
}

static void testSeededModelInitialization()
{
    const Model first{8, 1234u};
    const Model second{8, 1234u};
    const Model different{8, 5678u};

    requireTrue(
        first.W0.data == second.W0.data,
        "equal seeds should produce equal W0"
    );

    requireTrue(
        first.W1.data == second.W1.data,
        "equal seeds should produce equal W1"
    );

    requireTrue(
        first.W0.data != different.W0.data,
        "different seeds should produce different W0"
    );

    requireTrue(
        first.W1.data != different.W1.data,
        "different seeds should produce different W1"
    );
}


int main()
{
    testModelForward(1e-6f);
    testModelForwardIncorrectInputShape();
    testModelBackward(1e-6f);
    testModelBackwardNumerically(1e-3f);
    testSeededModelInitialization();

    std::cout << "All model tests passed\n";

    return 0;
}