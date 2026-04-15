// ModelNN for neural network models

#ifndef MODEL_NN_H
#define MODEL_NN_H

#include "Base.h"
#include "Matrix.h"


namespace ModelNN
{
	using Mat = Matrix::Matrix;

	// ModelNN::Model
	// Neural network model with layers:
	//  - input  (size 784)
	//  - hidden (size variable, default 128)
	//  - output (size 10)
	// Weight init based on sigmoid activation -> modify if using ReLu
	struct ModelNN
	{
		Size input{ 784 };
		Size output{ 10 };
		Size hidden{ 128 };
		Mat W0, b0, W1, b1;

		ModelNN() = default;
		
		explicit ModelNN(Size hiddenNodes = 128);

		Mat forward(Span<const Pixel> x) const;
		Mat backprop(Span<const Pixel> x) const;

		//
		void save(const String& path) const;
		static ModelNN load(const String& path);
		
	};


} // end namespace ModelNN

#endif // MODEL_NN_H