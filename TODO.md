TODO:

- matMul: currently accumulates into `out`; only works when `out` starts at zero.
    - Overwrite output or accumulate output? Lean towards make it overwrite by zeroing `out` internally before mult.
- tests:
    - matMul with known values
    - transposed matMul
    - sigmoid known values; `sigmoid(0) == 0.5`; `sigmoid(positive) > 0.5`
    - softmax sums to approx 1
    - cross-entropy with logits
- implement loss before gradients (use numerically stable form)
- avoid softmax in `ModelNN::forward`. Keep softmax in loss/evaluation.
- Auto diff: don't do full general autodiff; implement small matrix-level autodiff that supports
    - matMul
    - matAdd
    - matSigmoid
    - crossEntropyWithLogits
    - maybe matScale
