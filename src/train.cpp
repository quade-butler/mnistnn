#include "train.h"

#include <cmath>
#include <algorithm>

namespace mnistnn
{
	float crossEntropyLoss(const Matrix& logits, Label target)
	{
		// add input shape handling for `logits`, `target`
		
		float max_value{ *std::max_element(logits.data.begin(), logits.data.end()) };

		float sum{ 0.0f };
		for (Size i{ 0 }; i < logits.rows * logits.cols; ++i)
		{
			sum += std::exp(logits.data[i] - max_value);
		}
		
		// float loss { max_value + std::log(sum) - logits.data[static_cast<Size>(target)] };

		return max_value + std::log(sum) - logits.data[static_cast<Size>(target)];
	}
	void train(Model& model, const TrainConfig)
	{
		/*
		* 0. What inputs does train() need? Model? Config?
		* 1. Outer loop over epochs
		* 2. Inner loop over batches
		* 
		*  - shuffle training data
		*  - 
		*/
	}
} // end namespace TrainNN
