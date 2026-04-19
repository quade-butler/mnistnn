# TODO
- matMul: currently accumulates into `out`; only works when `out` starts at zero.
    - COMPLETE: made it zero out
- tests:
    COMPLETE:
    - cross-entropy with logits
    - matMul with known values
    - transposed matMul
    - sigmoid known values; `sigmoid(0) == 0.5`; `sigmoid(positive) > 0.5`
    - softmax sums to approx 1
- implement loss before gradients (use numerically stable form)
- avoid softmax in `ModelNN::forward`. Keep softmax in loss/evaluation.
    - COMPLETE
- Auto diff: don't do full general autodiff; implement small matrix-level autodiff that supports
    - matMul
    - matAdd
    - matSigmoid
    - crossEntropyWithLogits
    - maybe matScale

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