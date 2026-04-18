#include "dataset.h"

#include <fstream>
#include <cstdint>
#include <stdexcept>

#include <string>
#include <iostream>
#include <iomanip> // for std::setw() in write/print


namespace mnistnn
{
	namespace
	{
		// Swaps the endianness of a 32-bit integer
		static constexpr std::uint32_t byteSwap32(std::uint32_t x) 
		{
			return	((x & 0x000000FFu) << 24)|
					((x & 0x0000FF00u) << 8) |
					((x & 0x00FF0000u) >> 8) |
					((x & 0xFF000000u) >> 24);
		}

		// Reorients pixel data so that it is right-side up.
		// * NOTE: original EMNIST data is transposed/rotated.
		static void reorientData(
			Vector<Pixel>& img_pixels, 
			Size num_images, 
			Size dim_size) 
		{

			if (img_pixels.size() != num_images * dim_size * dim_size)
				throw std::runtime_error("ERROR: Image size mismatch");

			Vector<Pixel> transformed_img(num_images * dim_size * dim_size);
			Size total_pixels{ dim_size * dim_size };

			for (Size i{ 0 }; i < num_images; ++i) 
			{
				for (Size r{ 0 }; r < dim_size; ++r) 
				{
					for (Size c{ 0 }; c < dim_size; ++c) 
					{
						transformed_img[i * total_pixels + r * dim_size + c] =
							img_pixels[i * total_pixels + c * dim_size + r];
					}
				}
			}

			img_pixels.swap(transformed_img);
		}
	} // end namespace

	Dataset loadIdxDataset(
		const std::string& images_path, 
		const std::string& labels_path)
	{
		// Images
		std::ifstream images_file(images_path, std::ios::binary);

		if (!images_file)
			throw std::runtime_error("ERROR: Cannot read images file.");

		// allocate 2 bytes as per seekg() below
		Vector<Pixel> magic_header(2);

		// skip first 2 bytes
		images_file.seekg(2, std::ios::beg);
		images_file.read(
			reinterpret_cast<char*>(magic_header.data()), magic_header.size()
		);

		std::uint32_t num_images{};
		std::uint32_t num_rows{};
		std::uint32_t num_cols{};

		images_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
		images_file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
		images_file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

		// dimension sizes are big endian -> swap to little endian and cast
		num_images = static_cast<Size>(byteSwap32(num_images));
		num_rows   = static_cast<Size>(byteSwap32(num_rows));
		num_cols   = static_cast<Size>(byteSwap32(num_cols));

		Vector<Pixel> images(num_images * num_rows * num_cols);

		images_file.read(
			reinterpret_cast<char*>(images.data()),
			static_cast<std::streamsize>(images.size()));

		reorientData(images, num_images, num_rows);

		images_file.close();

		// Labels
		std::ifstream labels_file(labels_path, std::ios::binary);

		if (!labels_file)
			throw std::runtime_error("ERROR: Cannot read labels file.");

		// skip first 2 bytes
		labels_file.seekg(2, std::ios::beg);

		labels_file.read(
			reinterpret_cast<char*>(magic_header.data()), magic_header.size()
		);

		if (magic_header.back() != 1)
			throw std::runtime_error(
				"ERROR: Second dimension of labels needs to == 1."
			);

		std::uint32_t num_labels{};

		labels_file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

		num_labels = static_cast<Size>(byteSwap32(num_labels));

		if (num_images != num_labels)
			throw std::runtime_error(
				"ERROR: Number of images doesn't match number of labels."
			);

		Vector<Label> labels(num_labels);

		labels_file.read(reinterpret_cast<char*>(labels.data()), labels.size());

		labels_file.close();

		return Dataset{ num_images, num_rows, num_cols, images, labels };
	}

	const Span<const Pixel> Dataset::image(Size i) const
	{
		const Size stride = rows * cols;
		return Span<const Pixel>(
			images.data() + i * stride,
			stride
		);
	}

	void Dataset::print(Size image_number) const
	{
		std::cout << "Image number: " << image_number << '\n';
		std::cout << "Label: " << static_cast<int>(labels[image_number]) << '\n';

		const Span<const Pixel> img = image(image_number);

		for (Size i{ 0 }; i < img.size(); ++i)
		{
			std::cout << std::setw(3) << static_cast<int>(img[i]) << ' ';
			if ((i + 1) % 28 == 0)
				std::cout << '\n';
		}
		std::cout << '\n';
	}

} // end namespace mnistnn
