#include "train.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "dataset.h"
#include "matrix.h"
#include "model.h"

namespace mnistnn
{
	float crossEntropyLoss(const Matrix& logits, Label target)
	{

		if ( !(logits.rows == 10 && logits.cols == 1) )
		{
			throw std::invalid_argument("Logits must be 10 x 1");
		}
		if ( !(logits.data.size() == logits.rows * logits.cols) )
		{
			throw std::invalid_argument("Logits data size must be consistent with its Matrix size");
		}
		if ( !(target < logits.rows) )
		{
			throw std::invalid_argument("Target must be a digit from 0 to 9");
		}
		if (logits.data.empty())
		{
			throw std::invalid_argument("Logits must be non-empty");
		}

		float max_value{ *std::max_element(logits.data.begin(), logits.data.end()) };

		float sum{ 0.0f };
		for (Size i{ 0 }; i < logits.rows * logits.cols; ++i)
		{
			sum += std::exp(logits.data[i] - max_value);
		}

		return std::log(sum) - (logits.data[static_cast<Size>(target)] - max_value);
	}

	Matrix crossEntropyGradLogits(const Matrix& logits, Label target)
	{
		if ( !(logits.rows == 10 && logits.cols == 1) )
		{
			throw std::invalid_argument("Logits must be 10 x 1");
		}
		if ( !(logits.data.size() == logits.rows * logits.cols) )
		{
			throw std::invalid_argument("Logits data size must be consistent with its Matrix size");
		}
		if ( !(target < logits.rows) )
		{
			throw std::invalid_argument("Target must be a digit from 0 to 9");
		}
		if (logits.data.empty())
		{
			throw std::invalid_argument("Logits must be non-empty");
		}

		Matrix gradient(linalg::softmax(logits));
		gradient.data[static_cast<Size>(target)] -= 1.0f;

		return gradient;
	}

	namespace
	{
		void updateParameter(Matrix& parameter, const Matrix& gradient, float learning_rate)
		{
			if (
				parameter.rows != gradient.rows || 
				parameter.cols != gradient.cols || 
				parameter.data.size() != gradient.data.size())
			{
				throw std::invalid_argument("updateParameter: parameter and gradient shapes differ");
			}

			for (Size i{ 0 }; i < parameter.data.size(); ++i)
			{
				parameter.data[i] -= learning_rate * gradient.data[i];
			}

		}
	} // end namespace

    void applyGradients(Model& model, const ModelGradients& gradients, float learning_rate)
    {
		if (std::isinf(learning_rate) || learning_rate < 0)
		{
			throw std::invalid_argument("applyGradients: learning rate must be positive and finite");
		}

		updateParameter(model.W0, gradients.dW0, learning_rate);
		updateParameter(model.b0, gradients.db0, learning_rate);
		updateParameter(model.W1, gradients.dW1, learning_rate);
		updateParameter(model.b1, gradients.db1, learning_rate);
    }

    TrainStats train(Model& model, const Dataset& data, const TrainConfig& config)
	{
		/*
		  - TrainCofig.batch must be 1
		  - TrainStates.loss and .acc describe the final epoch
		  - Loss and acc are measured before each sample's update
		  - shuffling occurs once per epoch when enabled
		*/
		
		if (config.epochs == 0 || config.batch != 1)
		{
			throw std::invalid_argument("train: epochs must be grater than 0 and batch must be 1");
		}
		if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0f)
		{
			throw std::invalid_argument("train: learning rate must be positive and finite");
		}
		if (data.n == 0)
		{
			throw std::invalid_argument("train: dataset must be non-empty");
		}
		if (data.labels.size() != data.n)
		{
			throw std::invalid_argument("train: dataset's number of labels must equal number of images");
		}
		if (data.images.size() != data.n * data.rows * data.cols)
		{
			throw std::invalid_argument("train: incorrect image count in dataset");
		}
		if (data.rows * data.cols != model.num_input)
		{
			throw std::invalid_argument("train: image dimensions do not equal number of model input nodes");
		}

		Vector<Size> indices(data.n);
		std::iota(indices.begin(), indices.end(), Size{ 0 });

		std::mt19937 generator{ config.seed };

		TrainStats stats{};

		for (Size epoch{ 0 }; epoch < config.epochs; ++epoch)
		{
			if (config.shuffle_epoch)
			{
				std::shuffle(indices.begin(), indices.end(), generator);
			}

			double total_loss{ 0.0 };
			Size correct{ 0 };

			for (const Size index : indices)
			{
				const Label target{ data.labels[index] };
				const ForwardCache cache{ model.forwardWithCache(data.image(index)) };

				total_loss += crossEntropyLoss(cache.logits, target);

				const auto predicted_it{
					std::max_element(cache.logits.data.begin(), cache.logits.data.end())
				};

				const Size predicted{
					static_cast<Size>(std::distance(cache.logits.data.begin(), predicted_it))
				};

				if (predicted == static_cast<Size>(target)) { ++correct; }

				const Matrix dlogits{ crossEntropyGradLogits(cache.logits, target) };
				const ModelGradients gradients{ model.backward(cache, dlogits) };

				applyGradients(model, gradients, config.learning_rate);
			}

			stats.loss = static_cast<float>(total_loss / data.n);
			stats.acc = static_cast<float>(correct) / static_cast<float>(data.n);
		}
		
		return stats;
	}
} // end namespace mnistnn
