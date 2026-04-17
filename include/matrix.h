// MATRIX

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
	void fillRandf(Matrix& out, float bound);

	void scale(Matrix& out, float scale);

	void matMul(
		Matrix& out, const Matrix& a, const Matrix& b,
		bool transpose_a=false, bool transpose_b=false
	);
	void add(Matrix& out, const Matrix& a, const Matrix& b);
	void sub(Matrix& out, const Matrix& a, const Matrix& b);
	
	void sigmoid(Matrix& out);
	void relu(Matrix& out);
	void softmax(Matrix& out);

	void print(const Matrix& a);

} // end namespace mnistnn

#endif // MATRIX_H