// TrainNN for neural network training

#ifndef TRAIN_NN_H
#define TRAIN_NN_H

#include "base.h"
#include "matrix.h"
#include "model.h"
#include "dataset.h"

// TrainNN::Train needs to:
//	- provide a train() function
//	- provide a shuffle() function to shuffle data-label pairs
namespace mnistnn
{
	using Matrix	= linalg::Matrix;

	struct TrainConfig
	{
		Size  epochs{ 5 };
		Size  batch{ 64 };
		float rate{ 0.01f };

		bool shuffle_epoch{ true };
	};

	struct TrainStats {
		float train_loss{ 0.0f };
		float train_acc{ 0.0f };
		float test_loss{ 0.0f };
		float test_acc{ 0.0f };
	};

	TrainStats train(
		Model& model,
		const Dataset& train_data, 
		const TrainConfig& config
	);
	
	void shuffle();

	float crossEntropyLoss(const Matrix& logits, Label target);
} // end namespace mnistnn

#endif // TRAIN_NN_H