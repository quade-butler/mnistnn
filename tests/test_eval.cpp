#include "eval.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "matrix.h"

using mnistnn::Dataset;
using mnistnn::EvalStats;
using mnistnn::Label;
using mnistnn::Model;
using mnistnn::Pixel;
using mnistnn::Vector;
using mnistnn::evaluate;
using mnistnn::linalg::fill;

static void requireNear(
    float actual,
    float expected,
    float tolerance,
    const char* message
)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        std::cerr << message << ": expected " << expected
                  << ", got " << actual << '\n';
        throw std::runtime_error("test failed");
    }
}

static void testEvaluateKnownResult()
{
    Model model(2);

    fill(model.W0, 0.0f);
    fill(model.b0, 0.0f);
    fill(model.W1, 0.0f);
    fill(model.b1, 0.0f);

    Vector<Pixel> images(2 * model.num_input, 0);
    Vector<Label> labels{0, 1};

    Dataset dataset{
        2,
        28,
        28,
        images,
        labels
    };

    const EvalStats stats{evaluate(model, dataset)};

    requireNear(stats.loss, std::log(10.0f), 1e-6f,
                "evaluation loss");
    requireNear(stats.acc, 0.5f, 1e-6f,
                "evaluation accuracy");
}

int main()
{
    testEvaluateKnownResult();

    std::cout << "All evaluation tests passed\n";
}