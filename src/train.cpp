#include "base.h"
#include "matrix.h"
#include "train.h"

#include <cmath>
#include <algorithm>

namespace mnistnn
{
	float crossEntropyLoss(const Mat& logits, Pixel y)
	{
		// add input shape handling for `logits`
		
		float max_value{ *std::max_element(logits.data.begin(), logits.data.end()) };

		float sum{ 0.0f };
		for (Size i{ 0 }; i < logits.rows * logits.cols; ++i)
		{
			sum += std::exp(logits.data[i] - max_value);
		}
		
		// loss = max_value + std::log();

		// return loss;
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
