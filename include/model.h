// ModelNN for neural network models

#ifndef MODEL_NN_H
#define MODEL_NN_H

#include "base.h"
#include "matrix.h"

#include <string>

namespace mnistnn
{
	using Matrix = linalg::Matrix;

	struct ForwardCache
	{
		Matrix x;
		Matrix z1;
		Matrix a1;
		Matrix logits;
	};

	// mnistnn::Model
	// Neural network model with layers:
	//  - input  (size 784)
	//  - hidden (size variable, default 128)
	//  - output (size 10)
	// Weight init based on sigmoid activation -> modify if using ReLu
	class Model
	{
	public:
		Size num_input{ 784 };
		Size num_output{ 10 };
		Size num_hidden{ 128 };
		Matrix W0, b0, W1, b1;

		Model() = default;
		
		explicit Model(Size hidden_nodes = 128);

		// 
		Matrix forward(Span<const Pixel> x) const;
		ForwardCache forwardWithCache(Span<const Pixel> x) const;

		Matrix backprop(Span<const Pixel> x) const;

		//
		void save(const std::string& path) const;
		
		static Model load(const std::string& path);
	};

} // end namespace mnistnn

#endif // MODEL_NN_H