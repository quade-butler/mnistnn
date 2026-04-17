#include "base.h"
#include "matrix.h"

#include <cstdint>		// std::uint32_t
#include <random>		// PRNG inside matFillRandf()
#include <algorithm>	// std::max_element
#include <cmath>		// transcendental functions

#include <iostream>
#include <iomanip> // for std::setw() in write/print


namespace mnistnn::linalg
{

	void fill(Matrix& out, float value)
	{
		for (Size i{ 0 }; i < out.data.size(); ++i)
			out.data[i] = value;
	}

	// Fills matrix with uniformly distributed values between [-bound, bound]
	void fillRandf(Matrix& out, float bound)
	{
		static thread_local std::mt19937 gen{ std::random_device{}() };
		std::uniform_real_distribution<float> dis(-bound, bound);

		for (Size i{ 0 }; i < out.data.size(); ++i)
			out.data[i] = dis(gen);
	}

	void scale(Matrix& out, float scale)
	{
		for (Size i{ 0 }; i < out.rows * out.cols; ++i)
		{
			out.data[i] *= scale;
		}
	}

	namespace
	{
		void matMulnn(Matrix& out, const Matrix& a, const Matrix& b)
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

		void matMulnt(Matrix& out, const Matrix& a, const Matrix& b)
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

		void matMultn(Matrix& out, const Matrix& a, const Matrix& b)
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
		
		void matMultt(Matrix& out, const Matrix& a, const Matrix& b)
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
	} // end namespace

	// matMul:
	// `out` accumulates with `+=`, so `out` must start zeroed out.
	// A consequence is that you cannot call `matMul(a, a, b)`
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
			throw std::invalid_argument("ERROR: a_cols != b_rows");
		if (out.rows != a_rows || out.cols != b_cols)
			throw std::invalid_argument("ERROR: incorrect size of out matrix");

		// zero out
		fill(out, 0.0f);

		std::uint32_t transpose{ 
			static_cast<std::uint32_t>((transpose_a << 1) | (transpose_b << 0))
		};
		switch (transpose)
		{
		case 0b00:
			matMulnn(out, a, b);
			break;
		case 0b01:
			matMulnt(out, a, b);
			break;
		case 0b10:
			matMultn(out, a, b);
			break;
		case 0b11:
			matMultt(out, a, b);
			break;
		}
	}

	void add(Matrix& out, const Matrix& a, const Matrix& b)
	{
		if (a.cols != b.cols || a.rows != b.rows)
			throw std::runtime_error(
				"ERROR: Dimensions of A and B do not match"
			);
		if (out.cols != a.cols || out.rows != a.rows)
			throw std::runtime_error("ERROR: Dim(OUT) != DIM(A)");

		Size size{ out.cols * out.rows };
		for (Size i{ 0 }; i < size; ++i)
		{
			out.data[i] = a.data[i] + b.data[i];
		}
	}

	void sub(Matrix& out, const Matrix& a, const Matrix& b)
	{
		if (a.cols != b.cols || a.rows != b.rows)
			throw std::runtime_error(
				"ERROR: Dimensions of A and B do not match"
			);
		if (out.cols != a.cols || out.rows != a.rows)
			throw std::runtime_error("ERROR: Dim(OUT) != DIM(A)");

		Size size{ out.cols * out.rows };
		for (Size i{ 0 }; i < size; ++i)
		{
			out.data[i] = a.data[i] - b.data[i];
		}
	}

	void sigmoid(Matrix& out)
	{
		for (Size i{ 0 }; i < out.rows * out.cols; ++i)
		{
			out.data[i] = 1 / (1 + std::exp(-out.data[i]));
		}
	}

	void relu(Matrix& out)
	{

	}

	void softmax(Matrix& out)
	{
		float maxVal{ *std::max_element(out.data.begin(), out.data.end()) };
		float sum{ 0.0f };
		for (Size i{ 0 }; i < out.rows * out.cols; ++i)
		{
			out.data[i] = std::exp(out.data[i] - maxVal);
			sum += out.data[i];
		}

		scale(out, 1.0f / sum);
	}

	void print(const Matrix& a)
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

} // end namespace mnistnn
