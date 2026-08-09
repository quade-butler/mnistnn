#ifndef TRAIN_H
#define TRAIN_H

#include <cstdint>

#include "base.h"
#include "dataset.h"
#include "matrix.h"
#include "model.h"

namespace mnistnn
{
	using Matrix	= linalg::Matrix;

	struct TrainConfig
	{
		Size  epochs{ 5 };
		Size  batch{ 64 };
		std::uint32_t seed{ 1234u };
		float learning_rate{ 0.01f };
		bool shuffle_epoch{ true };
	};

	struct TrainStats {
		float loss{ 0.0f };
		float acc{ 0.0f };
	};

	TrainStats train(
		Model& model,
		const Dataset& train_data, 
		const TrainConfig& config
	);

	float crossEntropyLoss(const Matrix& logits, Label target);
	Matrix crossEntropyGradLogits(const Matrix& logits, Label target);
	void applyGradients(
		Model& model, const ModelGradients& gradients, float learning_rate
	);
} // end namespace mnistnn

#endif // TRAIN_H