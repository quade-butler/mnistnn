# mnistnn

`mnistnn` is a small C++20 neural-network project for learning matrix operations, forward and backward propagation, mini-batch gradient descent, evaluation, and model checkpointing without a machine learning framework.

The current model has:

- 784 input values for a flattened `28 x 28` image
- one configurable sigmoid hidden layer
- 10 output logits, one for each digit class
- cross-entropy loss with softmax computed during loss and evaluation

## Requirements

- CMake 3.20 or newer
- A C++20 compiler
- MNIST IDX image and label files

## Data files

Place these files in `data/`:

```text
data/
  train-images-idx3-ubyte.bin
  train-labels-idx1-ubyte.bin
  test-images-idx3-ubyte.bin
  test-labels-idx1-ubyte.bin
```

The data directory is compiled into the executable by CMake. The `.bin` files are ignored by Git.

## Build and test

Configure a Debug build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

For longer training experiments, use an optimized build:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

### Compare matrix-multiplication loop orders

The project can be compiled with a naive `i-j-k` implementation, fixed
cache-aware loop orders, or the default shape-aware `AUTO` selection:

```bash
cmake -S . -B build-naive \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNISTNN_MATMUL_IMPLEMENTATION=NAIVE

cmake -S . -B build-optimized \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNISTNN_MATMUL_IMPLEMENTATION=OPTIMIZED

cmake --build build-naive
cmake --build build-optimized
```

`AUTO` uses a dedicated matrix-vector kernel for one-column non-transposed
outputs, selects `i-j-k` for two or three columns, and selects `i-k-j` for four
or more columns. It also uses a dedicated outer-product kernel when an
`A x B-transposed` multiplication has a shared dimension of one. Configure it
explicitly with:

```bash
cmake -S . -B build-auto \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNISTNN_MATMUL_IMPLEMENTATION=AUTO
cmake --build build-auto
```

Run the same seeded training command with both executables and compare their
reported `Training time`. Dataset loading, evaluation, and checkpoint writing
are outside the timed region.

For a controlled cache-locality experiment, build both variants with GCC at
`-O2` while disabling compiler loop interchange, vectorization, unrolling,
peeling, and loop versioning:

```bash
cmake -S . -B build-naive-controlled \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNISTNN_MATMUL_IMPLEMENTATION=NAIVE \
  -DMNISTNN_CONTROLLED_CACHE_BENCHMARK=ON

cmake -S . -B build-optimized-controlled \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNISTNN_MATMUL_IMPLEMENTATION=OPTIMIZED \
  -DMNISTNN_CONTROLLED_CACHE_BENCHMARK=ON

cmake -S . -B build-auto-controlled \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNISTNN_MATMUL_IMPLEMENTATION=AUTO \
  -DMNISTNN_CONTROLLED_CACHE_BENCHMARK=ON

cmake --build build-naive-controlled
cmake --build build-optimized-controlled
cmake --build build-auto-controlled
```

This controlled mode applies only to `mnistnn_core`, where the matrix and
training operations are compiled. It currently requires GCC. Keep the normal
`-O3` builds as a separate measurement of production performance.

Run the standalone matrix benchmark from both controlled builds:

```bash
./build-naive-controlled/benchmark_matmul
./build-optimized-controlled/benchmark_matmul
./build-auto-controlled/benchmark_matmul
```

The benchmark multiplies a `128 x 784` matrix by matrices with `1`, `2`, `4`,
`8`, `16`, `32`, `64`, and `128` columns. It reports time per multiplication
and GFLOP/s, allowing the matrix-vector case, the loop-order crossover, and
progressively denser matrix-matrix cases to be compared without training-loop
overhead. It also measures the `128 x 1` by transposed `784 x 1` outer product
used to compute the first-layer weight gradient.

## Command-line usage

Show the built-in help:

```bash
./build/mnistnn --help
```

The application provides `train` and `eval` commands:

```text
mnistnn train <checkpoint> [options]
mnistnn eval <checkpoint> [options]
```

### Train a model

Train with the default settings and save the resulting model:

```bash
./build/mnistnn train build/model.bin
```

The training command accepts:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--epochs N` | `20` | Number of passes over the selected training data |
| `--batch N` | `32` | Number of sample gradients averaged per update |
| `--learning-rate RATE` | `0.1` | Gradient-descent learning rate |
| `--train-count N` | `500` | Number of training samples loaded from the start of the dataset |
| `--test-count N` | `200` | Number of test samples used during evaluation |
| `--hidden N` | `32` | Number of hidden-layer nodes |
| `--seed N` | `1234` | Seed for weight initialization and epoch shuffling |

Example with explicit settings:

```bash
./build/mnistnn train build/model.bin \
  --epochs 30 \
  --batch 32 \
  --learning-rate 0.2 \
  --train-count 1000 \
  --test-count 500 \
  --hidden 64 \
  --seed 1234
```

Training prints loss and accuracy before and after training, then writes the checkpoint path supplied on the command line. Accuracy is printed as a proportion between `0` and `1`.

The checkpoint's parent directory must already exist.

### Evaluate a checkpoint

Evaluate a saved model using the default first 200 test samples:

```bash
./build/mnistnn eval build/model.bin
```

Choose a different number of test samples with:

```bash
./build/mnistnn eval build/model.bin --test-count 1000
```

`eval` accepts only `--test-count`. The model architecture is restored from the checkpoint.

## Development workflow

A typical edit cycle is:

```bash
cmake --build build
ctest --test-dir build --output-on-failure
./build/mnistnn train build/model.bin
./build/mnistnn eval build/model.bin
```

Build a single test target when working in one area:

```bash
cmake --build build --target test_matrix
./build/test_matrix
```

Available test executables are:

- `test_matrix`
- `test_model`
- `test_train`
- `test_eval`

## Current limitations

- Mini-batches accumulate per-sample gradients; forward and backward passes are not vectorized across a batch.
- Training and evaluation operate on a prefix of each dataset rather than a randomized subset.
- The custom checkpoint format is intended for local use. It stores native binary integers and floats and is not guaranteed to be portable across architectures.
- The command-line program creates a new model for every `train` command; resuming an existing checkpoint is not yet exposed as a CLI option.
