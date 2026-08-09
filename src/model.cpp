#include "model.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "matrix.h"

namespace mnistnn
{
	using Matrix = linalg::Matrix;

	Model::Model(Size hidden_nodes)
		: Model(
			hidden_nodes,static_cast<std::uint32_t>(std::random_device{}())
		)
	{}

	Model::Model(Size hidden_nodes, std::uint32_t seed)
		: num_hidden{hidden_nodes},
		W0{num_hidden, num_input},
		b0{num_hidden, 1},
		W1{num_output, num_hidden},
		b1{num_output, 1}
	{
		if (hidden_nodes == 0)
		{
			throw std::invalid_argument("Model: hidden node count must be positive");
		}

		std::mt19937 generator{seed};

		const float bound_0{
			std::sqrt(6.0f / static_cast<float>(num_input + num_hidden))
		};

		const float bound_1{
			std::sqrt(6.0f / static_cast<float>(num_hidden + num_output))
		};

		fillRandf(W0, bound_0, generator);
		fillRandf(W1, bound_1, generator);
	}

	namespace 
	{
		// Converts Span<Pixel> pixels to Matrix float, then normalizes
		Matrix inputToMatrix(Span<const Pixel> x)
		{
			Matrix out(x.size(), 1);

			for (Size i{ 0 }; i < x.size(); ++i)
			{
				out.data[i] = static_cast<float>(x[i]) / 255.0f;
			}

			return out;
		}

		constexpr std::array<char, 8> model_magic{
			'M', 'N', 'I', 'S', 'T', 'N', 'N', '\0'
		};

		constexpr std::uint32_t model_format_version{1};

		template<typename T>
		void writeValue(std::ostream& output, const T& value)
		{
			static_assert(std::is_trivially_copyable_v<T>);

			output.write(
				reinterpret_cast<const char*>(&value),
				static_cast<std::streamsize>(sizeof(T))
			);

			if (!output)
			{
				throw std::runtime_error("Model::save: failed to write checkpoint");
			}
		}

		template<typename T>
		T readValue(std::istream& input)
		{
			static_assert(std::is_trivially_copyable_v<T>);

			T value{};

			input.read(
				reinterpret_cast<char*>(&value),
				static_cast<std::streamsize>(sizeof(T))
			);

			if (!input)
			{
				throw std::runtime_error("Model::load: truncated or unreadable checkpoint");
			}

			return value;
		}

		void writeMatrix(std::ostream& output, const Matrix& matrix)
		{
			if (matrix.rows != 0 &&
				matrix.cols >
					std::numeric_limits<Size>::max() / matrix.rows)
			{
				throw std::runtime_error("Model::save: matrix dimensions overflow");
			}

			if (matrix.data.size() != matrix.rows * matrix.cols)
			{
				throw std::runtime_error("Model::save: invalid matrix shape");
			}

			writeValue(output, static_cast<std::uint64_t>(matrix.rows));
			writeValue(output, static_cast<std::uint64_t>(matrix.cols));
			writeValue(output, static_cast<std::uint64_t>(matrix.data.size()));

			const std::size_t byte_count{ matrix.data.size() * sizeof(float) };

			output.write(
				reinterpret_cast<const char*>(matrix.data.data()),
				static_cast<std::streamsize>(byte_count)
			);

			if (!output)
			{
				throw std::runtime_error("Model::save: failed to write matrix data");
			}
		}

		void readMatrix(std::istream& input, Matrix& matrix)
		{
			const std::uint64_t rows{readValue<std::uint64_t>(input)};
			const std::uint64_t cols{readValue<std::uint64_t>(input)};
			const std::uint64_t count{readValue<std::uint64_t>(input)};

			if (rows != static_cast<std::uint64_t>(matrix.rows) ||
				cols != static_cast<std::uint64_t>(matrix.cols) ||
				count !=static_cast<std::uint64_t>(matrix.data.size())
			)
			{
				throw std::runtime_error("Model::load: matrix shape does not match model");
			}

			const std::size_t byte_count{ matrix.data.size() * sizeof(float) };

			input.read(
				reinterpret_cast<char*>(matrix.data.data()),
				static_cast<std::streamsize>(byte_count)
			);

			if (!input)
			{
				throw std::runtime_error("Model::load: truncated matrix data");
			}
		}
	}  // end namespace

	Matrix Model::forward(Span<const Pixel> x) const
	{
		ForwardCache cache { forwardWithCache(x) };
		return cache.logits;
	}

	ForwardCache Model::forwardWithCache(Span<const Pixel> x) const
	{
		if (x.size() != num_input)
		{
			std::cerr << "Expected 784 pixels. Got " << x.size() << '\n';
			throw std::invalid_argument("ERROR: Invalid argument.");
		}
		
		Matrix x_float{ inputToMatrix(x) };
		
		Matrix z1(num_hidden, 1);
		matMul(z1, W0, x_float);
		add(z1, z1, b0);
		Matrix a1{ sigmoid(z1) };

		Matrix z2(num_output, 1);
		matMul(z2, W1, a1);
		add(z2, z2, b1);
		
		return ForwardCache {x_float, z1, a1, z2};
	}

	ModelGradients Model::backward(const ForwardCache& cache, const Matrix& dlogits) const
	{
		const Matrix& x = cache.x;
		const Matrix& a1 = cache.a1;

		Matrix dW0(num_hidden, num_input);
		Matrix db0(num_hidden, 1);
		Matrix dW1(num_output, num_hidden);
		Matrix db1(num_output, 1);

		matMul(dW1, dlogits, a1, false, true);
		db1 = dlogits;

		Matrix da1(num_hidden, 1);
		matMul(da1, W1, dlogits, true, false);

		Matrix dz1(num_hidden, 1);
		for (Size i{ 0 }; i < dz1.data.size(); ++i)
		{
			dz1.data[i] = da1.data[i] * a1.data[i] * (1.0f - a1.data[i]);
		}

		matMul(dW0, dz1, x, false, true);
		db0 = dz1;

		ModelGradients gradients{dW0, db0, dW1, db1};

		return gradients;
	}

	void Model::save(const std::string& path) const
	{
		if (num_input == 0 || num_hidden == 0 || num_output == 0)
		{
			throw std::runtime_error("Model::save: model dimensions must be positive");
		}

		if (W0.rows != num_hidden ||
			W0.cols != num_input ||
			b0.rows != num_hidden ||
			b0.cols != 1 ||
			W1.rows != num_output ||
			W1.cols != num_hidden ||
			b1.rows != num_output ||
			b1.cols != 1)
		{
			throw std::runtime_error("Model::save: parameter shapes do not match model");
		}

		std::ofstream output{path, std::ios::binary | std::ios::trunc};

		if (!output)
		{
			throw std::runtime_error("Model::save: cannot open file: " + path);
		}

		output.write(model_magic.data(),static_cast<std::streamsize>(model_magic.size()));

		if (!output)
		{
			throw std::runtime_error("Model::save: failed to write file header");
		}

		writeValue(output, model_format_version);

		writeValue(output, static_cast<std::uint64_t>(num_input));
		writeValue(output, static_cast<std::uint64_t>(num_hidden));
		writeValue(output, static_cast<std::uint64_t>(num_output));

		writeMatrix(output, W0);
		writeMatrix(output, b0);
		writeMatrix(output, W1);
		writeMatrix(output, b1);

		output.flush();

		if (!output)
		{
			throw std::runtime_error("Model::save: failed to finalize checkpoint");
		}
	}

	Model Model::load(const std::string& path)
	{
		std::ifstream input{path, std::ios::binary};

		if (!input)
		{
			throw std::runtime_error("Model::load: cannot open file: " + path);
		}

		std::array<char, model_magic.size()> file_magic{};

		input.read(
			file_magic.data(), static_cast<std::streamsize>(file_magic.size())
		);

		if (!input)
		{
			throw std::runtime_error("Model::load: truncated file header");
		}

		if (file_magic != model_magic)
		{
			throw std::runtime_error("Model::load: invalid checkpoint identifier");
		}

		const std::uint32_t version{
			readValue<std::uint32_t>(input)
		};

		if (version != model_format_version)
		{
			throw std::runtime_error("Model::load: unsupported checkpoint version");
		}

		const std::uint64_t input_nodes{ readValue<std::uint64_t>(input) };
		const std::uint64_t hidden_nodes{ readValue<std::uint64_t>(input) };
		const std::uint64_t output_nodes{ readValue<std::uint64_t>(input) };

		if (input_nodes != 784)
		{
			throw std::runtime_error("Model::load: unsupported input size");
		}

		if (output_nodes != 10)
		{
			throw std::runtime_error("Model::load: unsupported output size");
		}

		if (hidden_nodes == 0 ||
			hidden_nodes > static_cast<std::uint64_t>(std::numeric_limits<Size>::max())
			)
		{
			throw std::runtime_error("Model::load: invalid hidden size");
		}

		Model model{ static_cast<Size>(hidden_nodes), 0u };

		readMatrix(input, model.W0);
		readMatrix(input, model.b0);
		readMatrix(input, model.W1);
		readMatrix(input, model.b1);

		return model;
	}

} // end namespace mnistnn
