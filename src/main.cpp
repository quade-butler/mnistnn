#include "base.h"
#include "matrix.h"
#include "dataset.h"
#include "model.h"

#include <string>
#include <iostream>

#define PRINT(var) (std::cout << var << '\n')

int main()
{
    using namespace mnistnn;
    using Matrix = linalg::Matrix;
    
    // Image data paths
    // Note: original data was EMNIST (transposed/rotated) not MNIST
    const std::string data_dir{ MNISTNN_DATA_DIR };
    // const std::string train_images{ data_dir + "train-images-idx3-ubyte.bin" };
    // const std::string train_labels{ data_dir + "train-labels-idx1-ubyte.bin" };
    const std::string test_images { data_dir + "test-images-idx3-ubyte.bin"  };
    const std::string test_labels { data_dir + "test-labels-idx1-ubyte.bin"  };

    // const Dataset train{ loadIdxDataset(train_images,train_labels) };
    const Dataset test{ loadIdxDataset(test_images, test_labels) };

    Model model{128};
    
    Matrix z = model.forward(test.image(0));

    print(z);

    return 0;
}

