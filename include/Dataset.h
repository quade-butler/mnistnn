// DATASET
// Reads in MNIST images and labels.

#ifndef DATASET_H
#define DATASET_H

#include "Base.h"

namespace Dataset 
{
	struct Dataset 
	{
		Size n{ 0 };	 // num images
		Size rows{ 28 }; // num rows
		Size cols{ 28 }; // num cols

		Vector<Pixel> images;
		Vector<Pixel> labels;


		// Returns an image given an index
		const Span<const Pixel> image(Size i) const;

		// Prints image & label to terminal
		void print(Size imgNum) const;
	};

	Dataset loadIdxDataset(const String& imagesPath, const String& labelsPath);

} // end namespace Dataset

#endif // DATASET_H