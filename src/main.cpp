#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <stdexcept>
#include <system_error>

#include "base.h"
#include "eval.h"
#include "dataset.h"
#include "matrix.h"
#include "model.h"
#include "train.h"

struct TrainOptions
{
    std::string checkpoint_path;

    mnistnn::Size epochs{ 20 };
    mnistnn::Size batch{ 32 };
    mnistnn::Size train_count{ 500 };
    mnistnn::Size test_count{ 200 };
    mnistnn::Size hidden_nodes{ 32 };

    float learning_rate{ 0.1f };
    std::uint32_t seed{ 1234u };
};

struct EvalOptions
{
    std::string checkpoint_path;
    mnistnn::Size test_count{ 200 };
};

template<typename T>
T parseNumber(std::string_view text, std::string_view option_name)
{
    T value{};

    const char* first{ text.data() };
    const char* last{ text.data() + text.size() };
    const auto result{ std::from_chars(first, last, value) };

    if (result.ec != std::errc{} || result.ptr != last)
    {
        throw std::invalid_argument(
            "invalid value for " 
            + std::string(option_name)
            + ": " 
            + std::string(text)
        );
    }

    return value;
}

TrainOptions parseTrainOptions(int argc, char* argv[])
{
    if (argc < 3)
    {
        throw std::invalid_argument("train requires a checkpoint path");
    }

    TrainOptions options{};
    options.checkpoint_path = argv[2];

    for (int i{3}; i < argc; i += 2)
    {
        if (i + 1 >= argc)
        {
            throw std::invalid_argument(
                "missing value for option: " + std::string(argv[i])
            );
        }

        const std::string_view name{argv[i]};
        const std::string_view value{argv[i + 1]};

        if (name == "--epochs")
        {
            options.epochs = parseNumber<mnistnn::Size>(value, name);
        }
        else if (name == "--batch")
        {
            options.batch = parseNumber<mnistnn::Size>(value, name);
        }
        else if (name == "--learning-rate")
        {
            options.learning_rate = parseNumber<float>(value, name);
        }
        else if (name == "--train-count")
        {
            options.train_count = parseNumber<mnistnn::Size>(value, name);
        }
        else if (name == "--test-count")
        {
            options.test_count = parseNumber<mnistnn::Size>(value, name);
        }
        else if (name == "--hidden")
        {
            options.hidden_nodes = parseNumber<mnistnn::Size>(value, name);
        }
        else if (name == "--seed")
        {
            options.seed = parseNumber<std::uint32_t>(value, name);
        }
        else
        {
            throw std::invalid_argument(
                "unknown train option: " + std::string(name)
            );
        }
    }

    if (options.epochs == 0)
    {
        throw std::invalid_argument("--epochs must be greater than zero");
    }

    if (options.batch == 0)
    {
        throw std::invalid_argument("--batch must be greater than zero");
    }

    if (options.train_count == 0)
    {
        throw std::invalid_argument("--train-count must be greater than zero");
    }

    if (options.test_count == 0)
    {
        throw std::invalid_argument("--test-count must be greater than zero");
    }

    if (options.hidden_nodes == 0)
    {
        throw std::invalid_argument("--hidden must be greater than zero");
    }

    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0f)
    {
        throw std::invalid_argument("--learning-rate must be positive and finite");
    }

    return options;
}

EvalOptions parseEvalOptions(int argc, char* argv[])
{
    if (argc < 3)
    {
        throw std::invalid_argument("eval requires a checkpoint path");
    }

    EvalOptions options{};
    options.checkpoint_path = argv[2];

    for (int i{3}; i < argc; i += 2)
    {
        if (i + 1 >= argc)
        {
            throw std::invalid_argument(
                "missing value for option: " + std::string(argv[i])
            );
        }

        const std::string_view name{ argv[i] };
        const std::string_view value{ argv[i + 1] };

        if (name == "--test-count")
        {
            options.test_count = parseNumber<mnistnn::Size>(value, name);
        }
        else
        {
            throw std::invalid_argument(
                "unknown eval option: " + std::string(name)
            );
        }
    }

    if (options.test_count == 0)
    {
        throw std::invalid_argument("--test-count must be greater than zero");
    }

    return options;
}

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

    void printUsage()
    {
        std::cerr << "Usage:\n"
                  << "  mnistnn train <checkpoint> [options]\n"
                  << "  mnistnn eval <checkpoint> [options]\n\n"
                  << "Train options:\n"
                  << "  --epochs N\n"
                  << "  --batch N\n"
                  << "  --learning-rate RATE\n"
                  << "  --train-count N\n"
                  << "  --test-count N\n"
                  << "  --hidden N\n"
                  << "  --seed N\n\n"
                  << "Eval options:\n"
                  << "  --test-count N\n";
    }
}

int runTrain(const TrainOptions& options)
{
    using namespace mnistnn;

    constexpr Size train_count{ 1000 };
    constexpr Size test_count{ 500 };
    constexpr std::uint32_t experiment_seed{ 1234u };

    const std::string data_dir{ MNISTNN_DATA_DIR };

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

    const Dataset train_data{ prefix(complete_train, options.train_count) };

    const Dataset test_data{ prefix(complete_test, options.test_count) };

    Model model{ options.hidden_nodes, options.seed };

    printStats("Train before",evaluate(model, train_data));
    printStats("Test before",evaluate(model, test_data));

    TrainConfig config{};
    config.epochs = options.epochs;
    config.batch = options.batch;
    config.learning_rate = options.learning_rate;
    config.shuffle_epoch = true;
    config.seed = options.seed;

    std::cout << "Configuration:"
              << " epochs=" << options.epochs
              << " batch=" << options.batch
              << " learning_rate=" << options.learning_rate
              << " train_count=" << options.train_count
              << " test_count=" << options.test_count
              << " hidden=" << options.hidden_nodes
              << " seed=" << options.seed
              << '\n';

    std::cout << "Training...\n";

    const auto training_start{ std::chrono::steady_clock::now() };

    const TrainStats training{ train(model, train_data, config) };

    const auto training_end{ std::chrono::steady_clock::now() };
    const std::chrono::duration<double> training_duration{
        training_end - training_start
    };

    std::cout << "Final training epoch"
              << ": loss=" << training.loss
              << ", accuracy=" << training.acc
              << '\n';

    std::cout << "Training time: "
              << training_duration.count()
              << " seconds\n";

    printStats("Train after", evaluate(model, train_data));
    printStats("Test after", evaluate(model, test_data));

    model.save(options.checkpoint_path);

    std::cout << "Saved checkpoint to "
              << options.checkpoint_path
              << '\n';

    return 0;
}

int runEval(const EvalOptions& options)
{
    using namespace mnistnn;

    constexpr Size test_count{ 5000 };

    const std::string data_dir{ MNISTNN_DATA_DIR };

    std::cout << "Loading test dataset...\n";

    const Dataset complete_test{
        loadIdxDataset(
            data_dir + "test-images-idx3-ubyte.bin",
            data_dir + "test-labels-idx1-ubyte.bin"
        )
    };

    const Dataset test_data{ prefix(complete_test, options.test_count) };

    std::cout << "Loading checkpoint from " << options.checkpoint_path << '\n';

    const Model model{ Model::load(options.checkpoint_path) };

    printStats("Test", evaluate(model, test_data));

    return 0;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return 1;
    }

    const std::string_view command{argv[1]};

    if (command == "--help" || command == "-h")
    {
        printUsage();
        return 0;
    }

    try
    {
        if (command == "train")
        {
            return runTrain(parseTrainOptions(argc, argv));
        }

        if (command == "eval")
        {
            return runEval(parseEvalOptions(argc, argv));
        }

        std::cerr << "Unknown command: " << command << '\n';

        printUsage();
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "mnistnn: " << error.what() << '\n';
        return 1;
    }
}
