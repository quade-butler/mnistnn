# TODO
- mnistnn::Matrix can have its data changed to arbitrary shape despite the designed shape from (rows, cols).
- Auto diff: don't do full general autodiff; implement small matrix-level autodiff that supports
    - matMul
    - matAdd
    - matSigmoid
    - crossEntropyWithLogits
    - maybe matScale

# COMPLETE
- matMul: currently accumulates into `out`; only works when `out` starts at zero.
- avoid softmax in `ModelNN::forward`. Keep softmax in loss/evaluation.
- implement loss before gradients (use numerically stable form)
- tests:
    - cross-entropy with logits
    - matMul with known values
    - transposed matMul
    - sigmoid known values; `sigmoid(0) == 0.5`; `sigmoid(positive) > 0.5`
    - softmax sums to approx 1

# Building/Running
- cmake -S . -B build
- cmake --build build
- ctest --test-dir build --output-on-failure

# Build only test exe
- cmake --build build --target <test_name>

# Debugging
- cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
- cmake --build build
- ctest --test-dir build --output-on-failure
- Debug with VS code like normal