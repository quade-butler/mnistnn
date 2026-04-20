#include "matrix.h"

#include <algorithm>  // std::max_element
#include <cmath>	  // transcendental functions
#include <cstdint>	  // std::uint32_t
#include <iomanip>    // std::setw() in write/print
#include <iostream>
#include <random>	  // PRNG inside matFillRandf()

#include "base.h"

namespace mnistnn::linalg
{

	void fill(Matrix& out, float value)
	{
		for (Size i{ 0 }; i < out.data.size(); ++i)
			out.data[i] = value;
	}

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

	Matrix sigmoid(const Matrix& x)
	{
		Matrix out(x.rows, x.cols);
		
		for (Size i{ 0 }; i < x.rows * x.cols; ++i)
		{
			out.data[i] = 1 / (1 + std::exp(-x.data[i]));
		}

		return out;
	}

	Matrix relu(const Matrix& x)
	{
		Matrix out(x.rows, x.cols);
		for (Size i{ 0 }; i < x.rows * x.cols; ++i)
		{
			float value { x.data[i] };
			out.data[i] = (value > 0) ? value : 0.0f;
		}
		return out;
	}

	Matrix softmax(const Matrix& x)
	{
		Matrix out(x.rows, x.cols);
		
		float max_value{ *std::max_element(x.data.begin(), x.data.end()) };
		float sum{ 0.0f };
		for (Size i{ 0 }; i < x.rows * x.cols; ++i)
		{
			out.data[i] = std::exp(x.data[i] - max_value);
			sum += out.data[i];
		}

		scale(out, 1.0f / sum);

		return out;
	}

	void print(const Matrix& x)
	{
		for (Size i{ 0 }; i < x.rows; ++i)
		{
			std::cout << ' ';
			for (Size j{ 0 }; j < x.cols; ++j)
			{
				std::cout << std::setw(12)<< x.data[i * x.cols + j] << ' ';
			}
			std::cout << '\n';
		}
	}

} // end namespace mnistnn
