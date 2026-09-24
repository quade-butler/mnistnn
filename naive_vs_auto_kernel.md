# Matrix Multiplication Kernels: NAIVE, OPTIMIZED, and AUTO

## Overview

All three implementations calculate the same mathematical result:

```text
C = A * B
```

They differ in the order in which they visit the matrices and in whether they
recognize important special cases. This affects memory access, cache use,
register reuse, and the amount of general-purpose work performed around the
arithmetic.

The important result from this project is that a loop order is not inherently
"optimized" for every matrix shape. The first fixed loop-order optimization
improved conventional matrix-matrix multiplication, but made the narrow
matrix-vector operations used by the training loop slower. The `AUTO` version
uses the operation's shape to select a suitable kernel instead.

## Matrix storage and loop notation

`Matrix` stores its values in row-major order. Elements next to each other in a
row are contiguous in memory:

```text
A(i, k) = a.data[i * a.cols + k]
B(k, j) = b.data[k * b.cols + j]
C(i, j) = out.data[i * out.cols + j]
```

The names `i-j-k` and `i-k-j` describe the loops from outermost to innermost.
The innermost loop generally matters most for locality because it performs the
most frequent memory accesses.

## 1. NAIVE: fixed `i-j-k`

For a normal, non-transposed multiplication, NAIVE uses:

```text
for each output row i
    for each output column j
        for each shared-dimension value k
            C(i,j) += A(i,k) * B(k,j)
```

Conceptually, it finishes one output element before moving to the next one.

### Strengths

- It is simple and makes a good correctness reference.
- `A(i,k)` is read sequentially across a row as `k` changes.
- The repeated accumulation into one `C(i,j)` can often remain in a register.
- It is particularly suitable when `C` has one column.

### Weaknesses for wider matrices

When `B` has several columns, advancing `k` while holding `j` fixed accesses:

```text
B(0,j), B(1,j), B(2,j), ...
```

These values are separated in memory by `B.cols` elements. This is a strided
access rather than a sequential walk through a row. It uses cache lines less
efficiently as `B` becomes wider. Rows of `A` are also revisited for every
output column.

NAIVE therefore works reasonably well for narrow output, but scales poorly for
wider matrix-matrix multiplication.

## 2. OPTIMIZED: fixed cache-oriented loop orders

The first optimization changed the loop order according to the transpose mode,
but did not consider the dimensions of a particular call:

| Multiplication | OPTIMIZED loop order |
|---|---:|
| `A * B` | `i-k-j` |
| `A * B^T` | `i-j-k` |
| `A^T * B` | `k-i-j` |
| `A^T * B^T` | `i-j-k` |

For normal `A * B`, `i-k-j` is:

```text
for each output row i
    for each shared-dimension value k
        for each output column j
            C(i,j) += A(i,k) * B(k,j)
```

With `j` innermost, both `B(k,j)` and `C(i,j)` are accessed sequentially. One
value of `A(i,k)` is also reused across an entire output row. This is normally
a substantial improvement when the output has enough columns.

### Why it made training slower

The training loop processes one sample at a time, even when gradients are
accumulated into a mini-batch. Its important forward and backward shapes are:

| Operation | Shape with hidden size `H` | Category |
|---|---:|---|
| First-layer forward | `(H x 784) * (784 x 1)` | matrix-vector |
| Second-layer forward | `(10 x H) * (H x 1)` | matrix-vector |
| Output weight gradient | `(10 x 1) * (H x 1)^T` | outer product |
| Hidden gradient | `(10 x H)^T * (10 x 1)` | transposed matrix-vector |
| Input weight gradient | `(H x 1) * (784 x 1)^T` | outer product |

The `i-k-j` improvement targets a wide `j` dimension. For both forward
operations, however, `out.cols == 1`, so the `j` loop runs exactly once:

```text
for i
    for k
        C(i,0) += A(i,k) * B(k,0)
```

It gains none of the intended row-wise reuse. Instead of completing one dot
product with a local accumulator, it repeatedly performs a read-modify-write
of the same output element as `k` changes. The corresponding NAIVE traversal
is naturally a dot product:

```text
for i
    sum = 0
    for k
        sum += A(i,k) * B(k,0)
    C(i,0) = sum
```

The fixed `k-i-j` traversal for `A^T * B` has a similar shape mismatch in the
hidden-gradient calculation: the output again has only one column. It updates
the output incrementally across `k` rather than finishing one output value at a
time.

Meanwhile, the two large weight-gradient operations are outer products with a
shared dimension of one. The first OPTIMIZED implementation still sent these
through the general `i-j-k` kernel. It therefore did not accelerate the most
important shape-specific opportunity in backpropagation.

In short, the initial implementation optimized the wrong dimension for this
workload:

- Wide matrix-matrix multiplication improved.
- One-column matrix-vector multiplication became slower.
- Outer products remained general-purpose multiplications.
- Training contains many more narrow operations than the benchmark shape that
  motivated `i-k-j`.

This is why better cache locality in the abstract did not translate into a
faster end-to-end training loop.

## The microbenchmark showed the crossover

The controlled benchmark multiplies a `128 x 784` matrix by matrices with a
varying number of columns. Before adding the specialized AUTO kernels, the
representative timings were:

| Output columns | NAIVE `i-j-k` (ms) | OPTIMIZED `i-k-j` (ms) | Faster method |
|---:|---:|---:|---|
| 1 | 0.100 | 0.228 | NAIVE |
| 2 | 0.208 | 0.226 | NAIVE |
| 4 | 0.432 | 0.260 | OPTIMIZED |
| 8 | 0.781 | 0.455 | OPTIMIZED |
| 16 | 1.629 | 0.808 | OPTIMIZED |
| 32 | 3.385 | 1.541 | OPTIMIZED |
| 64 | 6.564 | 3.275 | OPTIMIZED |
| 128 | 13.186 | 6.600 | OPTIMIZED |

The crossover is between two and four output columns. A single fixed loop
order cannot capture both sides of that crossover.

These measurements are machine- and compiler-specific. Their important lesson
is the trend and the crossover, not the exact timing values.

## 3. AUTO: shape-aware kernel selection

AUTO keeps both general loop orders and dispatches according to the transpose
mode and actual dimensions:

| Operation or shape | AUTO selection |
|---|---|
| Normal output with one column | Dedicated matrix-vector kernel |
| Normal output with two or three columns | `i-j-k` |
| Normal output with four or more columns | `i-k-j` |
| `A * B^T` with shared dimension one | Dedicated outer-product kernel |
| General `A^T * B` | `k-i-j` |
| Other transpose cases | Existing appropriate general kernel |

The threshold of four columns comes from the measured crossover rather than
from an assumption that one loop order is always superior.

### Dedicated matrix-vector kernel

For a one-column normal multiplication, AUTO explicitly uses a scalar
accumulator:

```text
for i
    sum = 0
    for k
        sum += A(i,k) * B(k,0)
    C(i,0) = sum
```

This matches the favorable behavior of `i-j-k` while removing the one-iteration
`j` loop. It also overwrites every output element, so `matMul()` can skip the
separate zero-fill required by the accumulating general kernels.

### Dedicated outer-product kernel

When multiplying `(M x 1)` by the transpose of `(N x 1)`, the shared dimension
is exactly one. The general triple loop reduces to:

```text
for i
    a_i = A(i,0)
    for j
        C(i,j) = a_i * B(j,0)
```

This kernel:

- removes the redundant one-iteration `k` loop;
- reuses `A(i,0)` across an output row;
- reads and writes contiguous values as `j` changes;
- assigns results directly instead of zeroing and then accumulating them.

This matters greatly because `dW0` is an `H x 784` outer product for every
training sample. In the controlled microbenchmark, the dedicated `128 x 1` by
`(784 x 1)^T` kernel improved from about `0.129 ms` to `0.032 ms`, roughly a
fourfold speedup for that operation.

### What AUTO does not change

AUTO does not alter the neural-network mathematics, floating-point complexity,
training algorithm, batching behavior, or model results. It is still a small
scalar matrix implementation; it simply chooses a better traversal for known
shapes. It also does not guarantee that every branch is optimal on every CPU.
The name means automatic selection among this project's kernels, not a full
BLAS-style optimizer.

## End-to-end result

For five controlled training runs, the observed times were:

| Run | NAIVE (s) | AUTO (s) |
|---:|---:|---:|
| 1 | 34.8777 | 24.3723 |
| 2 | 34.4233 | 23.8828 |
| 3 | 34.4594 | 23.5289 |
| 4 | 34.5362 | 23.6403 |
| 5 | 34.6247 | 23.8259 |
| Median | **34.5362** | **23.8259** |

This corresponds to:

- approximately **1.45x** the throughput; or
- approximately a **31% reduction** in elapsed training time.

The speedup is mainly due to matching kernels to the actual training shapes,
especially specializing the large first-layer weight-gradient outer product.
It is not evidence that `i-k-j` is generally worse than `i-j-k`; the wider
microbenchmark cases show the opposite. It demonstrates that workload shape
must be considered when selecting a loop order.

## Controlled-build methodology

The comparison builds use the same GCC optimization settings. Controlled mode
uses `-O2` and disables compiler transformations that could independently
change the loops, including loop interchange, vectorization, unrolling,
peeling, and loop versioning. Dataset loading, evaluation, and checkpoint
writing are outside the reported training-time region.

Using identical flags matters because an unrestricted optimizer might
interchange, vectorize, or otherwise transform one source loop into code that
no longer represents the intended experiment. The controlled builds isolate
the effects of the handwritten traversal and shape selection more clearly.

The controlled experiment should be interpreted as a cache-locality and kernel
selection comparison. Normal optimized builds should still be measured
separately when evaluating real production performance.

## Main lesson

Loop-order optimization is shape-dependent:

- `i-j-k` is effective for dot products and very narrow output.
- `i-k-j` is effective when the output is wide enough to benefit from
  sequential row access and reuse.
- A shared dimension of one is an outer product and deserves a simpler kernel.
- End-to-end performance depends on how often each shape occurs, not just on
  the fastest result for one conventional matrix-matrix benchmark.

AUTO improved training because it encoded those distinctions instead of
applying one supposedly optimal loop order to every call.
