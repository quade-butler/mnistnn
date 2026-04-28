#include "matrix.h"

#include <algorithm>  // std::max_element
#include <cmath>	  // transcendental functions
#include <cstdint>	  // std::uint32_t
#include <iomanip>    // std::setw() in write/print
#include <iostream>
#include <random>	  // PRNG inside matFillRandf()
#include <stdexcept>

#include "base.h"

namespace mnistnn::linalg
{
	namespace
	{
		bool hasValidShape(const Matrix& m) {
			return m.data.size() == m.rows * m.cols;
		}
	}

	void fill(Matrix& out, float value)
	{
		if (!hasValidShape(out))
		{
			std::cerr << "out matrix has data size " << out.data.size()
					  << ", but declared size is " << out.cols * out.rows;
			throw std::invalid_argument(
				"fill: out matrix data size does not match declared size"
			);
		}	

		for (auto& num : out.data)
			num = value;
	}

	void fillRandf(Matrix& out, float bound)
	{
		if (!hasValidShape(out))
		{
			std::cerr << "out matrix has data size " << out.data.size()
					  << ", but declared size is " << out.cols * out.rows;
			throw std::invalid_argument(
				"fillRandf: out matrix data size does not match declared size"
			);
		}

		static thread_local std::mt19937 gen{ std::random_device{}() };
		std::uniform_real_distribution<float> dis(-bound, bound);
		
		for (auto& num : out.data)
			num = dis(gen);
	}

	void scale(Matrix& out, float scale)
	{
		if (!hasValidShape(out))
		{
			std::cerr << "out matrix has data size " << out.data.size()
					  << ", but declared size is " << out.cols * out.rows;
			throw std::invalid_argument(
				"scale: out matrix data size does not match declared size"
			);
		}
		if (out.data.empty())
			throw std::invalid_argument("scale: cannot scale out matrix when empty");

		for (auto& num : out.data)
			num *= scale;
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

		if (!hasValidShape(out))
		{
			std::cerr << "out matrix has data size " << out.data.size()
					  << ", but declared size is " << out.cols * out.rows;
			throw std::invalid_argument(
				"MatMul: out matrix data size does not match declared size"
			);
		}
		if (a_cols != b_rows)
		{
			std::cerr << "Matrix multiply dimensions are invalid. Left is " 
					  << a_rows << "x" << a_cols << " and right is "
					  << b_rows << "x" << b_cols << '\n';
			throw std::invalid_argument("matMul: a_cols != b_rows");
		}
		if (out.rows != a_rows || out.cols != b_cols)
		{
			std::cerr << "Invalid out matrix shape. Out is "
					  << out.rows << "x" << out.cols << ", but needs to be"
					  << a_rows << "x" << b_cols << '\n';
			throw std::invalid_argument("matMul: invalid size of out matrix");
		}

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
		Size a_rows{ a.rows };
		Size a_cols{ a.cols };
		Size b_rows{ b.rows };
		Size b_cols{ b.cols };

		if (!hasValidShape(out))
		{
			std::cerr << "out matrix has data size " << out.data.size()
					  << ", but declared size is " << out.cols * out.rows;
			throw std::invalid_argument(
				"add: out matrix data size does not match declared size"
			);
		}
		if (a.data.empty() || b.data.empty())
		{
			throw std::invalid_argument(
				"add: cannot perform matrix addition with empty input matrix"
			);
		}
		if (a_cols != b_cols || a_rows != b_rows)
		{
			std::cerr << "Matrix dimensions don't match for addition. Left is "
					  << a_rows << "x" << a_cols << " and right is "
					  << b_rows << "x" << b_cols << '\n';
			throw std::invalid_argument(
				"add: dimensions of A and B do not match"
			);
		}
		if (out.cols != a_cols || out.rows != a_rows)
		{
			std::cerr << "Invalid out matrix shape. Out is "
					  << out.rows << "x" << out.cols << ", but needs to be"
					  << a_rows << "x" << b_cols << '\n';
			throw std::invalid_argument("add: Dim(OUT) != DIM(A)");
		}
		for (Size i{ 0 }; i < out.data.size(); ++i)
		{
			out.data[i] = a.data[i] + b.data[i];
		}
	}

	void sub(Matrix& out, const Matrix& a, const Matrix& b)
	{
		Size a_rows{ a.rows };
		Size a_cols{ a.cols };
		Size b_rows{ b.rows };
		Size b_cols{ b.cols };		

		if (!hasValidShape(out))
		{
			std::cerr << "out matrix has data size " << out.data.size()
					  << ", but declared size is " << out.cols * out.rows;
			throw std::invalid_argument(
				"sub: out matrix data size does not match declared size"
			);
		}
		if (a.data.empty() || b.data.empty())
		{
			throw std::invalid_argument(
				"sub: cannot perform matrix subtraction with empty input matrix"
			);
		}
		if (a_cols != b_cols || a_rows != b_rows)
		{
			std::cerr << "Matrix dimensions don't match for subtraction. Left is " 
					  << a_rows << "x" << a_cols << " and right is "
					  << b_rows << "x" << b_cols << '\n';
			throw std::invalid_argument("sub: dimensions of A and B do not match");
		}
		if (out.cols != a_cols || out.rows != a_rows)
		{	
			std::cerr << "Invalid out matrix shape. Out is "
					  << out.rows << "x" << out.cols << ", but needs to be"
					  << a_rows << "x" << b_cols << '\n';
			throw std::invalid_argument("sub: Dim(OUT) != DIM(A)");
		}

		for (Size i{ 0 }; i < out.data.size(); ++i)
		{
			out.data[i] = a.data[i] - b.data[i];
		}
	}

	Matrix sigmoid(const Matrix& x)
	{
		if (x.data.empty())
		{
			throw std::invalid_argument(
				"sigmoid: cannot compute Sigmoid using empty input matrix"
			);
		}

		Matrix out(x.rows, x.cols);
		
		for (Size i{ 0 }; i < out.data.size(); ++i)
		{
			out.data[i] = 1 / (1 + std::exp(-x.data[i]));
		}

		return out;
	}

	Matrix relu(const Matrix& x)
	{
		if (x.data.empty())
			throw std::invalid_argument(
				"relu: cannot compute ReLu using empty input matrix"
			);

		Matrix out(x.rows, x.cols);

		for (Size i{ 0 }; i < out.data.size(); ++i)
		{
			float value { x.data[i] };
			out.data[i] = (value > 0) ? value : 0.0f;
		}

		return out;
	}

	Matrix softmax(const Matrix& x)
	{
		if (x.data.empty())
			throw std::invalid_argument(
				"softmax: cannot compute Softmax using empty input matrix"
			);

		Matrix out(x.rows, x.cols);
		
		float max_value{ *std::max_element(x.data.begin(), x.data.end()) };
		float sum{ 0.0f };

		for (Size i{ 0 }; i < out.data.size(); ++i)
		{
			out.data[i] = std::exp(x.data[i] - max_value);
			sum += out.data[i];
		}

		scale(out, 1.0f / sum);

		return out;
	}

	void print(const Matrix& x)
	{
		if (x.data.empty())
		{
			std::cout << "print: Cannot print. Matrix is empty.";
			return void();
		}
		
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
