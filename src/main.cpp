#include <string>
#include <iostream>

#include "base.h"
#include "eval.h"
#include "dataset.h"
#include "matrix.h"
#include "model.h"
#include "train.h"

#define PRINT(var) (std::cout << var << '\n')

namespace
{
    mnistnn::Dataset prefix(
        const mnistnn::Dataset& source,
        mnistnn::Size count
    )
    {
        if (count > source.n)
        {
            throw std::invalid_argument(
                "prefix size exceeds dataset size"
            );
        }

        const mnistnn::Size pixel_count{
            count * source.rows * source.cols
        };

        mnistnn::Vector<mnistnn::Pixel> images(
            source.images.begin(),
            source.images.begin() + pixel_count
        );

        mnistnn::Vector<mnistnn::Label> labels(
            source.labels.begin(),
            source.labels.begin() + count
        );

        return mnistnn::Dataset{
            count,
            source.rows,
            source.cols,
            images,
            labels
        };
    }

    void printStats(
        const char* name,
        const mnistnn::EvalStats& stats
    )
    {
        std::cout << name
                  << ": loss=" << stats.loss
                  << ", accuracy=" << stats.acc
                  << '\n';
    }
}

int main()
{
    try
    {
        using namespace mnistnn;
        using Matrix = linalg::Matrix;

        const std::string data_dir{ MNISTNN_DATA_DIR };

        const Dataset complete_train{
            loadIdxDataset(
                data_dir + "train-images-idx3-ubyte.bin",
                data_dir + "train-labels-idx1-ubyte.bin"
            )
        };

        const Dataset complete_test{
            loadIdxDataset(
                data_dir + "test-images-idx3-ubyte.bin",
                data_dir + "test-labels-idx1-ubyte.bin"
            )
        };
        
        // Keep the experiment quick while the implementation is unoptimized.
        const Dataset train_data{prefix(complete_train, 500)};
        const Dataset test_data{prefix(complete_test, 200)};

        Model model{32};

        printStats("Train before", evaluate(model, train_data));
        printStats("Test before", evaluate(model, test_data));

        TrainConfig config{};
        config.epochs = 10;
        config.batch = 1;
        config.learning_rate = 0.05f;
        config.shuffle_epoch = true;
        config.seed = 1234u;

        const TrainStats training{train(model, train_data, config)};

        std::cout << "Final training epoch"
                << ": loss=" << training.loss
                << ", accuracy=" << training.acc
                << '\n';

        printStats("Train after", evaluate(model, train_data));
        printStats("Test after", evaluate(model, test_data));

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "mnistnn: " << error.what() << '\n';
        return 1;
    }
}

