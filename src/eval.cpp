#include "eval.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "train.h"

namespace mnistnn
{
    EvalStats evaluate(const Model& model, const Dataset& data)
    {
        if (data.n == 0)
        {
            throw std::invalid_argument("evaluate: dataset must be non-empty");
        }

        if (data.labels.size() != data.n)
        {
            throw std::invalid_argument("evaluate: label count does not match image count");
        }

        if (data.images.size() !=
            data.n * data.rows * data.cols)
        {
            throw std::invalid_argument("evaluate: image data size is inconsistent");
        }

        if (data.rows * data.cols != model.num_input)
        {
            throw std::invalid_argument("evaluate: image dimensions do not match model input");
        }

        double total_loss{ 0.0f };
        Size correct{ 0 };

        for (Size i{ 0 }; i < data.n; ++i)
        {
            const Matrix logits{
                model.forward(data.image(i))
            };

            total_loss +=
                crossEntropyLoss(logits, data.labels[i]);

            const auto predicted_it{
                std::max_element(
                    logits.data.begin(),
                    logits.data.end()
                )
            };

            const Size prediction{
                static_cast<Size>(
                    std::distance(
                        logits.data.begin(),
                        predicted_it
                    )
                )
            };

            if (prediction ==
                static_cast<Size>(data.labels[i]))
            {
                ++correct;
            }
        }

        return EvalStats{
            static_cast<float>(total_loss / data.n),
            static_cast<float>(correct) /
                static_cast<float>(data.n)
        };
    }

}