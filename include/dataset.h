// DATASET
// Reads in MNIST images and labels.

#ifndef DATASET_H
#define DATASET_H

#include "base.h"

namespace mnistnn
{
	class Dataset 
	{
	public:
		Size n{ 0 };	 // num images
		Size rows{ 28 }; // num rows
		Size cols{ 28 }; // num cols

		Vector<Pixel> images;
		Vector<Pixel> labels;

		Dataset(Size n, Size r, Size c, Vector<Pixel> img, Vector<Pixel> lbl)
			: n{n}, rows{r}, cols{c}, images{img}, labels{lbl} {}


		// Returns an image given an index
		const Span<const Pixel> image(Size i) const;

		// Prints image & label to terminal
		void print(Size imgNum) const;
	};

	Dataset loadIdxDataset(const String& imagesPath, const String& labelsPath);

} // end namespace mnistnn

#endif // DATASET_H