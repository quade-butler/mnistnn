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

		ModelGradients zeroGradients(const Model& model)
		{
			return ModelGradients{
				Matrix(model.num_hidden, model.num_input),
				Matrix(model.num_hidden, 1),
				Matrix(model.num_output, model.num_hidden),
				Matrix(model.num_output, 1)
			};
		}

		void addInPlace(Matrix& total, const Matrix& addition)
		{
			if (total.rows != addition.rows ||
				total.cols != addition.cols ||
				total.data.size() != addition.data.size())
			{
				throw std::invalid_argument(
					"addInPlace: matrix shapes differ"
				);
			}

			for (Size i{0}; i < total.data.size(); ++i)
			{
				total.data[i] += addition.data[i];
			}
		}

		void accumulateGradients(
			ModelGradients& total,
			const ModelGradients& addition
		)
		{
			addInPlace(total.dW0, addition.dW0);
			addInPlace(total.db0, addition.db0);
			addInPlace(total.dW1, addition.dW1);
			addInPlace(total.db1, addition.db1);
		}
	} // end namespace

    void applyGradients(Model& model, const ModelGradients& gradients, float learning_rate)
    {
		if (!std::isfinite(learning_rate) || learning_rate <= 0.0f)
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
		if (config.epochs == 0)
		{
			throw std::invalid_argument("train: epochs must be greater than 0");
		}
		if (config.batch == 0)
		{
			throw std::invalid_argument("train: batch size must be greater than 0");
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

		for (Size epoch{0}; epoch < config.epochs; ++epoch)
		{
			if (config.shuffle_epoch)
			{
				std::shuffle(indices.begin(), indices.end(), generator);
			}

			double total_loss{0.0};
			Size correct{0};

			for (Size batch_begin{0}; batch_begin < data.n; batch_begin += config.batch)
			{
				const Size actual_batch_size{
					std::min(config.batch, data.n - batch_begin)
				};
				
				const Size batch_end{ batch_begin + actual_batch_size };

				ModelGradients batch_gradients{ zeroGradients(model) };

				for (Size position{batch_begin};
					position < batch_end;
					++position)
				{
					const Size index{indices[position]};
					const Label target{data.labels[index]};

					const ForwardCache cache{ 
						model.forwardWithCache(data.image(index))
					};

					total_loss += crossEntropyLoss(cache.logits, target);

					const auto predicted_it{
						std::max_element(
							cache.logits.data.begin(), cache.logits.data.end()
						)
					};

					const Size predicted{
						static_cast<Size>(
							std::distance(cache.logits.data.begin(), predicted_it)
						)
					};

					if (predicted == static_cast<Size>(target)) {++correct;}

					const Matrix dlogits{
						crossEntropyGradLogits(cache.logits, target)
					};

					const ModelGradients sample_gradients{
						model.backward(cache, dlogits)
					};

					accumulateGradients(batch_gradients, sample_gradients);
				}

				applyGradients(
					model, 
					batch_gradients, 
					config.learning_rate / static_cast<float>(actual_batch_size)
				);
			}

			stats.loss = static_cast<float>(total_loss / data.n);
			stats.acc = static_cast<float>(correct) /static_cast<float>(data.n);
		}

		return stats;
	}
} // end namespace mnistnn
