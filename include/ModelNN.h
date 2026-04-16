// ModelNN for neural network models

#ifndef MODEL_NN_H
#define MODEL_NN_H

#include "Base.h"
#include "Matrix.h"

namespace mnistnn
{
	using Mat = linalg::Matrix;

	// ModelNN::Model
	// Neural network model with layers:
	//  - input  (size 784)
	//  - hidden (size variable, default 128)
	//  - output (size 10)
	// Weight init based on sigmoid activation -> modify if using ReLu
	class Model
	{
	public:
		Size input{ 784 };
		Size output{ 10 };
		Size hidden{ 128 };
		Mat W0, b0, W1, b1;

		Model() = default;
		
		explicit Model(Size hiddenNodes = 128);

		Mat forward(Span<const Pixel> x) const;
		Mat backprop(Span<const Pixel> x) const;

		//
		void save(const String& path) const;
		static Model load(const String& path);
		
	};


} // end namespace mnistnn

#endif // MODEL_NN_H