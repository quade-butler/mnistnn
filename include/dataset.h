// DATASET
// Reads in MNIST images and labels.

#ifndef DATASET_H
#define DATASET_H

#include "base.h"

#include <string>

namespace mnistnn
{
	class Dataset 
	{
	public:
		Size n{ 0 };	 // num images
		Size rows{ 28 }; // num rows
		Size cols{ 28 }; // num cols

		Vector<Pixel> images;
		Vector<Label> labels;

		Dataset(Size n, Size r, Size c, Vector<Pixel> img, Vector<Label> lbl)
			: n{n}, rows{r}, cols{c}, images{img}, labels{lbl} {}


		// Returns an image given an index
		const Span<const Pixel> image(Size i) const;

		// Prints image & label to terminal
		void print(Size imgNum) const;
	};

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
	Dataset loadIdxDataset(
		const std::string& images_path, 
		const std::string& labels_path
	);

} // end namespace mnistnn

#endif // DATASET_H