#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "base.h"
#include "matrix.h"

namespace
{
    using mnistnn::Size;
    using mnistnn::linalg::Matrix;
    using mnistnn::linalg::fillRandf;
    using mnistnn::linalg::matMul;

    constexpr Size left_rows{128};
    constexpr Size shared_dimension{784};
    constexpr double minimum_sample_seconds{0.5};

    struct Result
    {
        Size columns{};
        Size iterations{};
        double milliseconds_per_multiply{};
        double gigaflops{};
        double checksum{};
    };

    double runIterations(
        Matrix& out,
        const Matrix& a,
        const Matrix& b,
        Size iterations,
        bool transpose_a = false,
        bool transpose_b = false
    )
    {
        const auto start{std::chrono::steady_clock::now()};

        for (Size iteration{0}; iteration < iterations; ++iteration)
        {
            matMul(out, a, b, transpose_a, transpose_b);
        }

        const auto end{std::chrono::steady_clock::now()};
        return std::chrono::duration<double>(end - start).count();
    }

    Result benchmarkShape(Size columns)
    {
        Matrix a(left_rows, shared_dimension);
        Matrix b(shared_dimension, columns);
        Matrix out(left_rows, columns);

        std::mt19937 generator{1234u};
        fillRandf(a, 1.0f, generator);
        fillRandf(b, 1.0f, generator);

        // Warm instruction and data caches before measuring.
        for (Size warmup{0}; warmup < 5; ++warmup)
        {
            matMul(out, a, b);
        }

        Size iterations{1};
        double elapsed_seconds{};

        for (;;)
        {
            elapsed_seconds = runIterations(out, a, b, iterations);

            if (elapsed_seconds >= minimum_sample_seconds)
            {
                break;
            }

            if (iterations > std::numeric_limits<Size>::max() / 2)
            {
                throw std::runtime_error("benchmark iteration count overflow");
            }

            iterations *= 2;
        }

        double checksum{};
        for (const float value : out.data)
        {
            checksum += value;
        }

        const double seconds_per_multiply{
            elapsed_seconds / static_cast<double>(iterations)
        };

        const double floating_point_operations{
            2.0 * static_cast<double>(left_rows)
                * static_cast<double>(shared_dimension)
                * static_cast<double>(columns)
        };

        return Result{
            columns,
            iterations,
            seconds_per_multiply * 1000.0,
            floating_point_operations / seconds_per_multiply / 1.0e9,
            checksum
        };
    }

    Result benchmarkOuterProduct()
    {
        Matrix a(left_rows, 1);
        Matrix b(shared_dimension, 1);
        Matrix out(left_rows, shared_dimension);

        std::mt19937 generator{1234u};
        fillRandf(a, 1.0f, generator);
        fillRandf(b, 1.0f, generator);

        for (Size warmup{0}; warmup < 5; ++warmup)
        {
            matMul(out, a, b, false, true);
        }

        Size iterations{1};
        double elapsed_seconds{};

        for (;;)
        {
            elapsed_seconds = runIterations(
                out, a, b, iterations, false, true
            );

            if (elapsed_seconds >= minimum_sample_seconds)
            {
                break;
            }

            if (iterations > std::numeric_limits<Size>::max() / 2)
            {
                throw std::runtime_error("benchmark iteration count overflow");
            }

            iterations *= 2;
        }

        double checksum{};
        for (const float value : out.data)
        {
            checksum += value;
        }

        const double seconds_per_multiply{
            elapsed_seconds / static_cast<double>(iterations)
        };

        const double floating_point_operations{
            static_cast<double>(left_rows)
                * static_cast<double>(shared_dimension)
        };

        return Result{
            shared_dimension,
            iterations,
            seconds_per_multiply * 1000.0,
            floating_point_operations / seconds_per_multiply / 1.0e9,
            checksum
        };
    }
}

int main()
{
    constexpr std::array<Size, 8> column_counts{
        1, 2, 4, 8, 16, 32, 64, 128
    };

    std::cout << "Implementation: " << MNISTNN_MATMUL_NAME << '\n'
              << "A shape: " << left_rows << 'x' << shared_dimension << '\n'
              << "Controlled timing target: at least "
              << minimum_sample_seconds << " seconds per shape\n\n"
              << std::left
              << std::setw(10) << "B cols"
              << std::setw(14) << "Iterations"
              << std::setw(16) << "ms/multiply"
              << std::setw(12) << "GFLOP/s"
              << "Checksum\n";

    for (const Size columns : column_counts)
    {
        const Result result{benchmarkShape(columns)};

        std::cout << std::left
                  << std::setw(10) << result.columns
                  << std::setw(14) << result.iterations
                  << std::setw(16) << std::fixed << std::setprecision(6)
                  << result.milliseconds_per_multiply
                  << std::setw(12) << std::setprecision(3)
                  << result.gigaflops
                  << std::setprecision(6) << result.checksum
                  << '\n';
    }

    const Result outer{benchmarkOuterProduct()};

    std::cout << "\nOuter product: "
              << left_rows << "x1 * ("
              << shared_dimension << "x1)^T\n"
              << "Iterations: " << outer.iterations << '\n'
              << "ms/multiply: " << std::fixed << std::setprecision(6)
              << outer.milliseconds_per_multiply << '\n'
              << "GFLOP/s: " << std::setprecision(3)
              << outer.gigaflops << '\n'
              << "Checksum: " << std::setprecision(6)
              << outer.checksum << '\n';
}
