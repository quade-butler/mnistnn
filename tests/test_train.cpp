#include "base.h"
#include "matrix.h"
#include "train.h"

#include <iostream>

using mnistnn::linalg::Matrix;
using mnistnn::Label;

void testCrossEntropyLoss()
{
    Label target{ 1 };
    Matrix logit(10, 1);
    
}

int main()
{
    testCrossEntropyLoss();

    std::cout << "All training tests passed\n";
}