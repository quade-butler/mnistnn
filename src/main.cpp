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
    mnistnn::Dataset prefix(const mnistnn::Dataset& source,mnistnn::Size count)
    {
        if (count > source.n)
        {
            throw std::invalid_argument("prefix size exceeds dataset size");
        }

        const mnistnn::Size pixel_count{ count * source.rows * source.cols };

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

    void printStats(const char* name, const mnistnn::EvalStats& stats)
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

        constexpr std::uint32_t experiment_seed{1234u};

        Model model{32, experiment_seed};

        printStats("Train before", evaluate(model, train_data));
        printStats("Test before", evaluate(model, test_data));

        TrainConfig config{};
        config.seed = experiment_seed;
        config.epochs = 10;
        config.batch = 32;
        config.learning_rate = 0.1f;
        config.shuffle_epoch = true;

        const TrainStats training{train(model, train_data, config)};

        const EvalStats trained_stats{ evaluate(model, test_data) };

        const std::string checkpoint_path{ "build/mnistnn-model.bin" };

        model.save(checkpoint_path);

        const Model loaded{ Model::load(checkpoint_path) };

        const EvalStats loaded_stats{ evaluate(loaded, test_data) };

        printStats("Trained model", trained_stats);
        printStats("Loaded model", loaded_stats);

        const Matrix original_logits{ model.forward(test_data.image(0)) };

        const Matrix loaded_logits{ loaded.forward(test_data.image(0)) };

        if (original_logits.data != loaded_logits.data)
        {
            throw std::runtime_error("loaded model output differs from saved model");
        }

        const float loss_before_resume{ evaluate(loaded, train_data).loss };

        Model resumed{loaded};

        train(resumed, train_data, config);

        const float loss_after_resume{ evaluate(resumed, train_data).loss };

        if (loss_after_resume >= loss_before_resume)
        {
            std::cerr << "Warning: resumed training did not reduce loss\n";
        }

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "mnistnn: " << error.what() << '\n';
        return 1;
    }
}

