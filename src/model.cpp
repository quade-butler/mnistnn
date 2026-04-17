#include "base.h"
#include "model.h"
#include "matrix.h"

#include <cmath>

namespace mnistnn
{
	using Mat = linalg::Matrix;

	Model::Model(Size hiddenNodes) 
		: hidden{hiddenNodes}, W0{hidden, input}, b0{hidden, 1}
		, W1{output, hidden}, b1{output, 1}
	{
		float bound0{ std::sqrt(6.0f / (input + hidden)) };
		float bound1{ std::sqrt(6.0f / (hidden + output)) };

		// biases kept at zero

		fillRandf(W0, bound0);
		fillRandf(W1, bound1);
	}
	
	// Converts Span<uint8_t> pixels to Matrix float, then normalizes
	static Mat inputToMatrix(Span<const Pixel> x)
	{
		Mat out(x.size(), 1);

		for (Size i{ 0 }; i < x.size(); ++i)
		{
			out.data[i] = static_cast<float>(x[i]) / 255.0f;
		}

		return out;
	}

	// ModelNN::forward
	// 	Input:	image  (type span<Pixel>)
	// 	Output:	logits (type Matrix)
	Mat Model::forward(Span<const Pixel> xui8) const
	{
		Mat x{ inputToMatrix(xui8) };
		
		Mat z1(hidden, 1);
		matMul(z1, W0, x);
		add(z1, z1, b0);
		sigmoid(z1);

		Mat z2(output, 1);
		matMul(z2, W1, z1);
		add(z2, z2, b1);
		
		return z2; // logits
	}

} // end namespace mnistnn
