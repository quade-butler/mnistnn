#include "Base.h"
#include "Matrix.h"

//#include <stdexcept>
#include <cstdint>		// std::uint32_t
#include <random>		// PRNG inside matFillRandf()
#include <algorithm>	// std::max_element
#include <cmath>		// transcendental functions

#include <iostream>
#include <iomanip> // for std::setw() in write/print


namespace Matrix
{

	void matFill(Matrix& out, float value)
	{
		for (Size i{ 0 }; i < out.data.size(); ++i)
			out.data[i] = value;
	}

	// Fills matrix with uniformly distributed values between [-bound, bound]
	void matFillRandf(Matrix& out, float bound)
	{
		static thread_local std::mt19937 gen{ std::random_device{}() };
		std::uniform_real_distribution<float> dis(-bound, bound);

		for (Size i{ 0 }; i < out.data.size(); ++i)
			out.data[i] = dis(gen);
	}

	void matScale(Matrix& out, float scale)
	{
		for (Size i{ 0 }; i < out.rows * out.cols; ++i)
		{
			out.data[i] *= scale;
		}
	}

	static void _matMul_nn(Matrix& out, const Matrix& a, const Matrix& b)
	{
		for (Size i{ 0 }; i < out.rows; ++i)
		{
			for (Size k{ 0 }; k < a.cols; ++k)
			{
				for (Size j{ 0 }; j < out.cols; ++j)
				{
					out.data[i * out.cols + j]
						+= a.data[i * a.cols + k]
						*  b.data[k * b.cols + j];
				}
			}
		}
	}

	static void _matMul_nt(Matrix& out, const Matrix& a, const Matrix& b)
	{
		for (Size i{ 0 }; i < out.rows; ++i)
		{	
			for (Size j{ 0 }; j < out.cols; ++j)
			{
				for (Size k{ 0 }; k < a.cols; ++k)
				{
					out.data[i * out.cols + j]
						+= a.data[i * a.cols + k]
						*  b.data[k + j * b.cols];
				}
			}
		}
	}

	static void _matMul_tn(Matrix& out, const Matrix& a, const Matrix& b)
	{
		for (Size k{ 0 }; k < a.rows; ++k)
		{
			for (Size i{ 0 }; i < out.rows; ++i)
			{
				for (Size j{ 0 }; j < out.cols; ++j)
				{
					out.data[i * out.cols + j]
						+= a.data[i + k * a.cols]
						*  b.data[j + k * b.cols];
				}
			}
		}
	}
	
	static void _matMul_tt(Matrix& out, const Matrix& a, const Matrix& b)
	{
		for (Size i{ 0 }; i < out.rows; ++i)
		{
			for (Size j{ 0 }; j < out.cols; ++j)
			{
				for (Size k{ 0 }; k < a.rows; ++k)
				{
					out.data[i * out.cols + j]
						+= a.data[i + k * a.cols]
						*  b.data[k + j * b.cols];
				}
			}
		}
	}

	void matMul(
		Matrix& out, const Matrix& a, const Matrix& b, 
		bool transpose_a, bool transpose_b
	)
	{
		Size a_rows{ transpose_a ? a.cols : a.rows };
		Size a_cols{ transpose_a ? a.rows : a.cols };
		Size b_rows{ transpose_b ? b.cols : b.rows };
		Size b_cols{ transpose_b ? b.rows : b.cols };

		if (a_cols != b_rows) 
			throw runtime_error("ERROR: a_cols != b_rows");
		if (out.rows != a_rows || out.cols != b_cols)
			throw runtime_error("ERROR: incorrect size of out matrix");

		std::uint32_t transpose{ 
			static_cast<std::uint32_t>((transpose_a << 1) | (transpose_b << 0))
		};
		switch (transpose)
		{
		case 0b00:
			_matMul_nn(out, a, b);
			break;
		case 0b01:
			_matMul_nt(out, a, b);
			break;
		case 0b10:
			_matMul_tn(out, a, b);
			break;
		case 0b11:
			_matMul_tt(out, a, b);
			break;
		}
	}

	void matAdd(Matrix& out, const Matrix& a, const Matrix& b)
	{
		if (a.cols != b.cols || a.rows != b.rows)
			throw runtime_error("ERROR: Dimensions of A and B do not match");
		if (out.cols != a.cols || out.rows != a.rows)
			throw runtime_error("ERROR: Dim(OUT) != DIM(A)");

		Size size{ out.cols * out.rows };
		for (Size i{ 0 }; i < size; ++i)
		{
			out.data[i] = a.data[i] + b.data[i];
		}
	}

	void matSub(Matrix& out, const Matrix& a, const Matrix& b)
	{
		if (a.cols != b.cols || a.rows != b.rows)
			throw runtime_error("ERROR: Dimensions of A and B do not match");
		if (out.cols != a.cols || out.rows != a.rows)
			throw runtime_error("ERROR: Dim(OUT) != DIM(A)");

		Size size{ out.cols * out.rows };
		for (Size i{ 0 }; i < size; ++i)
		{
			out.data[i] = a.data[i] - b.data[i];
		}
	}

	void matSigmoid(Matrix& out)
	{
		for (Size i{ 0 }; i < out.rows * out.cols; ++i)
		{
			out.data[i] = 1 / (1 + std::exp(out.data[i]));
		}
	}

	void matSoftmax(Matrix& out)
	{
		float maxVal{ *std::max_element(out.data.begin(), out.data.end()) };
		float sum{ 0.0f };
		for (Size i{ 0 }; i < out.rows * out.cols; ++i)
		{
			out.data[i] = std::exp(out.data[i] - maxVal);
			sum += out.data[i];
		}

		matScale(out, 1.0f / sum);
	}

	void matPrint(const Matrix& a)
	{
		for (Size i{ 0 }; i < a.rows; ++i)
		{
			std::cout << ' ';
			for (Size j{ 0 }; j < a.cols; ++j)
			{
				std::cout << std::setw(12)<< a.data[i * a.cols + j] << ' ';
			}
			std::cout << '\n';
		}
	}

} // end namespace Matrix
