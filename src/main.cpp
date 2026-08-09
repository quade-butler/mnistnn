#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "base.h"
#include "eval.h"
#include "dataset.h"
#include "matrix.h"
#include "model.h"
#include "train.h"

namespace
{
    mnistnn::Dataset prefix(const mnistnn::Dataset& source,mnistnn::Size count)
    {
        if (count > source.n)
        {
            throw std::invalid_argument("prefix size exceeds dataset size");
        }

        const mnistnn::Size pixel_count{ count * source.rows * source.cols };

        mnistnn::Vector<mnistnn::Pixel> images(
            source.images.begin(), source.images.begin() + pixel_count
        );

        mnistnn::Vector<mnistnn::Label> labels(
            source.labels.begin(), source.labels.begin() + count
        );

        return mnistnn::Dataset{
            count, source.rows, source.cols, images, labels
        };
    }

    void printStats(const char* name, const mnistnn::EvalStats& stats)
    {
        std::cout << name
                  << ": loss=" << stats.loss
                  << ", accuracy=" << stats.acc
                  << '\n';
    }
}

int runTrain(const std::string& checkpoint_path)
{
    using namespace mnistnn;

    constexpr Size train_count{500};
    constexpr Size test_count{200};
    constexpr std::uint32_t experiment_seed{1234u};

    const std::string data_dir{MNISTNN_DATA_DIR};

    std::cout << "Loading datasets...\n";

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

    const Dataset train_data{ prefix(complete_train, train_count) };

    const Dataset test_data{ prefix(complete_test, test_count) };

    Model model{32, experiment_seed};

    printStats("Train before",evaluate(model, train_data));
    printStats("Test before",evaluate(model, test_data));

    TrainConfig config{};
    config.epochs = 20;
    config.batch = 32;
    config.learning_rate = 0.1f;
    config.shuffle_epoch = true;
    config.seed = experiment_seed;

    std::cout << "Training...\n";

    const TrainStats training{ train(model, train_data, config) };

    std::cout
        << "Final training epoch"
        << ": loss=" << training.loss
        << ", accuracy=" << training.acc
        << '\n';

    printStats("Train after", evaluate(model, train_data));
    printStats("Test after", evaluate(model, test_data));

    model.save(checkpoint_path);

    std::cout
        << "Saved checkpoint to "
        << checkpoint_path
        << '\n';

    return 0;
}

int runEval(const std::string& checkpoint_path)
{
    using namespace mnistnn;

    constexpr Size test_count{200};

    const std::string data_dir{MNISTNN_DATA_DIR};

    std::cout << "Loading test dataset...\n";

    const Dataset complete_test{
        loadIdxDataset(
            data_dir + "test-images-idx3-ubyte.bin",
            data_dir + "test-labels-idx1-ubyte.bin"
        )
    };

    const Dataset test_data{ prefix(complete_test, test_count) };

    std::cout
        << "Loading checkpoint from "
        << checkpoint_path
        << '\n';

    const Model model{ Model::load(checkpoint_path) };

    printStats("Test", evaluate(model, test_data));

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr
            << "Usage:\n"
            << "  mnistnn train <checkpoint>\n"
            << "  mnistnn eval <checkpoint>\n";

        return 1;
    }

    const std::string_view command{argv[1]};
    const std::string checkpoint_path{argv[2]};

    try
    {
        if (command == "train")
        {
            return runTrain(checkpoint_path);
        }

        if (command == "eval")
        {
            return runEval(checkpoint_path);
        }

        std::cerr << "Unknown command: " << command << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "mnistnn: " << error.what() << '\n';
        return 1;
    }
}

