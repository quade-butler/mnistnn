#include "base.h"
#include "train.h"

namespace mnistnn
{
	float crossEntropyLogit(const Mat& logits, Pixel y)
	{
		
		return 0.0f;
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
