#include "Base.h"
#include "Dataset.h"

#include <fstream>
#include <cstdint>
#include <stdexcept>

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
		// * NOTE: original EMNIST data is transposed and rotated.
		static void reorientData(
			Vector<Pixel>& imgPixels, 
			Size numImages, 
			Size dimSize) 
		{

			if (imgPixels.size() != numImages * dimSize * dimSize)
				throw std::runtime_error("ERROR: Image size mismatch");

			Vector<Pixel> trafoData(numImages * dimSize * dimSize);
			Size totalPx{ dimSize * dimSize };

			for (Size i{ 0 }; i < numImages; ++i) 
			{
				for (Size r{ 0 }; r < dimSize; ++r) 
				{
					for (Size c{ 0 }; c < dimSize; ++c) 
					{
						trafoData[i * totalPx + r * dimSize + c] =
							imgPixels[i * totalPx + c * dimSize + r];
					}
				}
			}

			imgPixels.swap(trafoData);
		}
	} // end namespace

	// Loads .idx binary data.
	// IDX files have a header consisting of a magic number, e.g.,
	//
	//  		  MAGIC	   |			 |			   |
	//  	0x 00 00 08 03 | 00 00 EA 60 | 00 00 00 1C | 00 00 00 1C
	//  	   -- -- -- -- | -- -- -- -- | -- -- -- -- | -- -- -- --
	//  bytes: 1  2  3  4  | 	DIM 1	 |	DIM 2	   |	DIM 3
	// 
	// The magic number is 4 bytes long and decoded as:
	// The first 2 bytes are always 0.
	// The third byte encodes the type of the data: 0x08 - unsigned byte.
	// The fourth byte encodes the number of dimensions of the array.
	// The sizes in each dimension are 4-byte (32-bit) integers (big-endian)
	Dataset loadIdxDataset(const String& imagesPath, const String& labelsPath)
	{
		// Images
		std::ifstream imagesFile(imagesPath, std::ios::binary);

		if (!imagesFile)
			throw std::runtime_error("ERROR: Cannot read images file.");

		// allocate 2 bytes as per seekg() below
		Vector<Pixel> magicHeader(2);

		// skip first 2 bytes
		imagesFile.seekg(2, std::ios::beg);
		imagesFile.read(
			reinterpret_cast<char*>(magicHeader.data()), magicHeader.size()
		);

		std::uint32_t numImages{};
		std::uint32_t numRows{};
		std::uint32_t numCols{};

		imagesFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
		imagesFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
		imagesFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

		// dimension sizes are big endian -> swap to little endian and cast
		numImages = static_cast<Size>(byteSwap32(numImages));
		numRows   = static_cast<Size>(byteSwap32(numRows));
		numCols	  = static_cast<Size>(byteSwap32(numCols));

		Vector<Pixel> images(numImages * numRows * numCols);

		imagesFile.read(
			reinterpret_cast<char*>(images.data()),
			static_cast<std::streamsize>(images.size()));

		reorientData(images, numImages, numRows);

		imagesFile.close();

		// Labels
		std::ifstream labelsFile(labelsPath, std::ios::binary);

		if (!labelsFile)
			throw std::runtime_error("ERROR: Cannot read labels file.");

		// skip first 2 bytes
		labelsFile.seekg(2, std::ios::beg);

		labelsFile.read(
			reinterpret_cast<char*>(magicHeader.data()), magicHeader.size()
		);

		if (magicHeader.back() != 1)
			throw std::runtime_error(
				"ERROR: Second dimension of labels needs to == 1."
			);

		std::uint32_t numLabels{};

		labelsFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

		numLabels = static_cast<Size>(byteSwap32(numLabels));

		if (numImages != numLabels)
			throw std::runtime_error(
				"ERROR: Number of images doesn't match number of labels."
			);

		Vector<Pixel> labels(numLabels);

		labelsFile.read(reinterpret_cast<char*>(labels.data()), labels.size());

		labelsFile.close();

		return Dataset{ numImages, numRows, numCols, images, labels };
	}

	const Span<const Pixel> Dataset::image(Size i) const
	{
		const Size stride = rows * cols;
		return Span<const Pixel>(
			images.data() + i * stride,
			stride
		);
	}

	void Dataset::print(Size imgNum) const
	{
		std::cout << "Image number: " << imgNum << '\n';
		std::cout << "Label: " << static_cast<int>(labels[imgNum]) << '\n';

		const Span<const Pixel> img = image(imgNum);

		for (Size i{ 0 }; i < img.size(); ++i)
		{
			std::cout << std::setw(3) << static_cast<int>(img[i]) << ' ';
			if ((i + 1) % 28 == 0)
				std::cout << '\n';
		}
		std::cout << '\n';
	}

} // end namespace mnistnn
