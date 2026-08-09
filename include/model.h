#ifndef MODEL_H
#define MODEL_H

#include <string>

#include "base.h"
#include "matrix.h"

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

	struct ModelGradients {
		Matrix dW0;
		Matrix db0;
		Matrix dW1;
		Matrix db1;
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

		
		Matrix forward(Span<const Pixel> x) const;
		
		ForwardCache forwardWithCache(Span<const Pixel> x) const;

		ModelGradients backward(
			const ForwardCache& cache, 
			const Matrix& dlogits
		) const;

		void save(const std::string& path) const;
		
		static Model load(const std::string& path);
	};

} // end namespace mnistnn

#endif // MODEL_H