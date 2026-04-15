// TrainNN for neural network training

#ifndef TRAIN_NN_H
#define TRAIN_NN_H

#include "Base.h"
#include "Matrix.h"
#include "Dataset.h"
#include "DataLoader.h"
#include "ModelNN.h"



/*
* TrainNN::Train needs to:
*	- provide a train() function
*	- provide a shuffle() function to shuffle data-label pairs
* 
*/
namespace TrainNN
{
	using Mat	= Matrix::Matrix;
	using Model = ModelNN::ModelNN;

	struct TrainConfig
	{
		Size  epochs{ 5 };
		Size  batch{ 64 };
		float rate{ 0.01f };

		bool shuffleEpoch{ true };
		//Size seed{ 12345 };
	};

	struct TrainStats {
		float trainLoss{ 0.0f };
		float trainAcc{ 0.0f };
		float testLoss{ 0.0f };
		float testAcc{ 0.0f };
	};

	void shuffle();
	void cost();
	float crossEntropyLogit(const Mat& logits, Pixel y);
	void train(Model& Model, const TrainConfig);

} // end namespace TrainNN

#endif // TRAIN_NN_H