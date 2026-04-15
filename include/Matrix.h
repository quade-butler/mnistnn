// MATRIX

#ifndef MATRIX_H
#define MATRIX_H

#include "Base.h"

namespace Matrix
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

	void matFill(Matrix& out, float value);
	void matFillRandf(Matrix& out, float bound);

	void matScale(Matrix& out, float scale);

	void matMul(
		Matrix& out, const Matrix& a, const Matrix& b,
		bool transpose_a=false, bool transpose_b=false
	);
	void matAdd(Matrix& out, const Matrix& a, const Matrix& b);
	void matSub(Matrix& out, const Matrix& a, const Matrix& b);
	
	void matSigmoid(Matrix& out);
	void matRelu(Matrix& out);
	void matSoftmax(Matrix& out);

	void matPrint(const Matrix& a);

} // end namespace Matrix

#endif // MATRIX_H