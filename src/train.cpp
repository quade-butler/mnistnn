#include "train.h"

#include <algorithm>
#include <cmath>
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
	}

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

    TrainStats train(Model&, const Dataset&, const TrainConfig&)
	{
		/*
		* 0. What inputs does train() need? Model? Config?
		* 1. Outer loop over epochs
		* 2. Inner loop over batches
		* 
		*  - shuffle training data
		*  - 
		*/

		throw std::logic_error("train: not implemented");
	}
} // end namespace TrainNN
