#ifndef EVAL_H
#define EVAL_H

#include "dataset.h"
#include "model.h"

namespace mnistnn
{
	struct EvalStats
	{
		float loss{ 0.0f };
		float acc{ 0.0f };
	};

	EvalStats evaluate(const Model& model, const Dataset& data);

} // end namespace mnistnn

#endif // EVAL_H