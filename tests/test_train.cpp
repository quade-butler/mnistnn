#include "train.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "base.h"
#include "eval.h"
#include "dataset.h"
#include "matrix.h"
#include "model.h"

using mnistnn::Vector;
using mnistnn::linalg::Matrix;
using mnistnn::linalg::fill;
using mnistnn::Label;
using mnistnn::Size;
using mnistnn::Pixel;
using mnistnn::crossEntropyLoss;
using mnistnn::crossEntropyGradLogits;
using mnistnn::Model;
using mnistnn::ModelGradients;
using mnistnn::ForwardCache;
using mnistnn::Dataset;
using mnistnn::TrainConfig;
using mnistnn::TrainStats;
using mnistnn::EvalStats;

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

static void testApplyGradients(float tol)
{
    Model model(2);

    fill(model.W0, 1.0f);
    fill(model.b0, 2.0f);
    fill(model.W1, 3.0f);
    fill(model.b1, 4.0f);

    ModelGradients gradients{
        Matrix(model.num_hidden, model.num_input),
        Matrix(model.num_hidden, 1),
        Matrix(model.num_output, model.num_hidden),
        Matrix(model.num_output, 1)
    };

    fill(gradients.dW0, 0.5f);
    fill(gradients.db0, -1.0f);
    fill(gradients.dW1, 2.0f);
    fill(gradients.db1, -0.25f);

    applyGradients(model, gradients, 0.1f);

    requireNear(model.W0.data[0], 0.95f, tol, "updated W0");
    requireNear(model.b0.data[0], 2.1f, tol, "updated b0");
    requireNear(model.W1.data[0], 2.8f, tol, "updated W1");
    requireNear(model.b1.data[0], 4.025f, tol, "updated b1");
}

static void testSingleStepReducesLoss()
{
    Model model(2);

    fill(model.W0, 0.0f);
    fill(model.b0, 0.0f);
    fill(model.W1, 0.0f);
    fill(model.b1, 0.0f);

    model.W0.data[0] = 0.2f;
    model.W0.data[1] = -0.1f;

    const Size second_hidden_row{model.num_input};
    model.W0.data[second_hidden_row] = -0.3f;
    model.W0.data[second_hidden_row + 1] = 0.4f;

    model.W1.data[0] = 0.3f;
    model.W1.data[1] = -0.2f;
    model.W1.data[2] = -0.1f;
    model.W1.data[3] = 0.4f;

    Vector<Pixel> input(model.num_input, 0);
    input[0] = 255;
    input[1] = 128;

    const Label target{1};

    const ForwardCache cache{model.forwardWithCache(input)};
    const float loss_before{
        crossEntropyLoss(cache.logits, target)
    };

    const Matrix dlogits{
        crossEntropyGradLogits(cache.logits, target)
    };
    const ModelGradients gradients{
        model.backward(cache, dlogits)
    };

    applyGradients(model, gradients, 0.1f);

    const float loss_after{
        crossEntropyLoss(model.forward(input), target)
    };

    requireTrue(
        loss_after < loss_before,
        "one gradient step should reduce loss"
    );
}

static void testTrainOverfitsOneExample()
{
    Model model(2);

    fill(model.W0, 0.0f);
    fill(model.b0, 0.0f);
    fill(model.W1, 0.0f);
    fill(model.b1, 0.0f);

    Vector<Pixel> images(model.num_input, 0);
    images[0] = 255;

    Vector<Label> labels{3};

    Dataset dataset{
        1,
        28,
        28,
        images,
        labels
    };

    const float loss_before{
        crossEntropyLoss(
            model.forward(dataset.image(0)),
            labels[0]
        )
    };

    TrainConfig config{};
    config.epochs = 20;
    config.batch = 1;
    config.learning_rate = 0.1f;
    config.shuffle_epoch = false;

    const TrainStats stats{
        train(model, dataset, config)
    };

    const float loss_after{
        crossEntropyLoss(
            model.forward(dataset.image(0)),
            labels[0]
        )
    };

    requireTrue(
        loss_after < loss_before,
        "training should reduce loss on one example"
    );

    requireTrue(
        std::isfinite(stats.loss),
        "reported training loss should be finite"
    );

    requireTrue(
        stats.acc >= 0.0f && stats.acc <= 1.0f,
        "reported accuracy should be a valid proportion"
    );
}

static void testTrainOverfitsTinyDataset()
{
    constexpr Size sample_count{3};

    Model model(3);

    fill(model.W0, 0.0f);
    fill(model.b0, 0.0f);
    fill(model.W1, 0.0f);
    fill(model.b1, 0.0f);

    // Give each hidden node a different input feature.
    model.W0.data[0 * model.num_input + 0] = 1.0f;
    model.W0.data[1 * model.num_input + 1] = 1.0f;
    model.W0.data[2 * model.num_input + 2] = 1.0f;

    Vector<Pixel> images(
        sample_count * model.num_input,
        Pixel{0}
    );

    images[0 * model.num_input + 0] = 255;
    images[1 * model.num_input + 1] = 255;
    images[2 * model.num_input + 2] = 255;

    Vector<Label> labels{0, 1, 2};

    Dataset dataset{
        sample_count,
        28,
        28,
        images,
        labels
    };

    TrainConfig config{};
    config.epochs = 500;
    config.batch = 1;
    config.learning_rate = 0.1f;
    config.shuffle_epoch = true;
    config.seed = 1234u;

    train(model, dataset, config);

    const EvalStats stats{evaluate(model, dataset)};

    requireNear(stats.acc, 1.0f, 1e-6f, "model should classify every tiny training example");
    requireTrue(stats.loss < 0.5f, "model should reach low loss on the tiny dataset"); 
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
    testApplyGradients(1e-6f);
    testSingleStepReducesLoss();
    testTrainOverfitsOneExample();
    testTrainOverfitsTinyDataset();

    std::cout << "All training tests passed\n";

    return 0;
}