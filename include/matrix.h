#ifndef MATRIX_H
#define MATRIX_H

#include "base.h"

namespace mnistnn::linalg
{
	// Workflow with Matrix typically involves:
	// 1. Create a Matrix with Matrix() constructor.
	// 2. Operate on that matrix, e.g. matFill, matMul, matAdd, etc.
	// NOTE: Matrix only handles floats
	struct Matrix
	{
		Size rows{ 0 };
		Size cols{ 0 };
		Vector<float> data;

		Matrix() = default;

		explicit Matrix(Size r, Size c) 
			: rows{r}, cols{c}, data(r * c) {}
	};

	void fill(Matrix& out, float value);

	// Fills matrix with uniformly distributed values between [-bound, bound]
	void fillRandf(Matrix& out, float bound);

	void scale(Matrix& out, float scale);

	// `out` accumulates with `+=`, so `out` must start zeroed out.
	// A consequence is that you cannot call `matMul(a, a, b)`
	void matMul(
		Matrix& out, const Matrix& a, const Matrix& b,
		bool transpose_a=false, bool transpose_b=false
	);

	void add(Matrix& out, const Matrix& a, const Matrix& b);

	void sub(Matrix& out, const Matrix& a, const Matrix& b);
	
	Matrix sigmoid(const Matrix& x);
	Matrix relu(const Matrix& x);
	Matrix softmax(const Matrix& x);

	void print(const Matrix& x);

} // end namespace mnistnn

#endif // MATRIX_H