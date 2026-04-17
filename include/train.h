// TrainNN for neural network training

#ifndef TRAIN_NN_H
#define TRAIN_NN_H

#include "base.h"
#include "matrix.h"
#include "dataset.h"
#include "data_loader.h"
#include "model.h"

// TrainNN::Train needs to:
//	- provide a train() function
//	- provide a shuffle() function to shuffle data-label pairs
namespace mnistnn
{
	using Mat	= linalg::Matrix;

	struct TrainConfig
	{
		Size  epochs{ 5 };
		Size  batch{ 64 };
		float rate{ 0.01f };

		bool shuffle_epoch{ true };
		//Size seed{ 12345 };
	};

	struct TrainStats {
		float train_loss{ 0.0f };
		float train_acc{ 0.0f };
		float test_loss{ 0.0f };
		float test_acc{ 0.0f };
	};

	void shuffle();
	void cost();
	float crossEntropyLogit(const Mat& logits, Pixel y);
	void train(Model& Model, const TrainConfig);

} // end namespace mnistnn

#endif // TRAIN_NN_H