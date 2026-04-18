#include "model.h"

#include <cmath>

namespace mnistnn
{
	using Matrix = linalg::Matrix;

	Model::Model(Size hidden_nodes) 
		: num_hidden{hidden_nodes}, W0{num_hidden, num_input}, b0{num_hidden, 1}
		, W1{num_output, num_hidden}, b1{num_output, 1}
	{
		float bound_0{ std::sqrt(6.0f / (num_input + num_hidden)) };
		float bound_1{ std::sqrt(6.0f / (num_hidden + num_output)) };

		// biases kept at zero

		fillRandf(W0, bound_0);
		fillRandf(W1, bound_1);
	}

	namespace 
	{
		// Converts Span<uint8_t> pixels to Matrix float, then normalizes
		static Matrix inputToMatrix(Span<const Pixel> x)
		{
			Matrix out(x.size(), 1);

			for (Size i{ 0 }; i < x.size(); ++i)
			{
				out.data[i] = static_cast<float>(x[i]) / 255.0f;
			}

			return out;
		}
	}  // end namespace

	Matrix Model::forward(Span<const Pixel> x_uint8) const
	{
		Matrix x{ inputToMatrix(x_uint8) };
		
		Matrix z1(num_hidden, 1);
		matMul(z1, W0, x);
		add(z1, z1, b0);
		Matrix a1 { sigmoid(z1) };

		Matrix z2(num_output, 1);
		matMul(z2, W1, z1);
		add(z2, z2, b1);
		
		return z2; // logits
	}

} // end namespace mnistnn
