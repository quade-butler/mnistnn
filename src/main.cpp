#include "Base.h"
#include "Matrix.h"
#include "Dataset.h"
#include "ModelNN.h"

#include <iostream>
#include <iomanip> // for std::setw() in write/print

#define PRINT(var) (std::cout << var << '\n')

int main()
{
    using Mat   = Matrix::Matrix;
    using Model = ModelNN::ModelNN;
    // Image data paths
    
    const String dataDir{ MNISTNN_DATA_DIR };
    // const String trainImages{ dataDir + "train-images-idx3-ubyte.bin" };
    // const String trainLabels{ dataDir + "train-labels-idx1-ubyte.bin" };
    const String testImages { dataDir + "test-images-idx3-ubyte.bin"  };
    const String testLabels { dataDir + "test-labels-idx1-ubyte.bin"  };

    // const Dataset::Dataset train{
    // Dataset::loadIdxDataset(trainImages,trainLabels) };
    const Dataset::Dataset test{
        Dataset::loadIdxDataset(testImages,testLabels) };

    Model model{128};
    
    Mat z = model.forward(test.image(0));

    matPrint(z);

    return 0;
}

