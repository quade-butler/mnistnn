// DATALOADER
// Loads images via an index with random shuffling
// Used when passing images to ModelNN.feedforward()

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "base.h"
#include "dataset.h"

namespace mnistnn
{

	class DataLoader
	{
	public:
		const Dataset& ds;
		Size pos{ 0 };
		Vector<Size> idx;

		explicit DataLoader(const Dataset& d) 
			: ds(d), idx(d.n)
		{
			for (Size i{ 0 }; i < d.n; ++i) idx[i] = i;
		}

		// shuffle idx with RNG for each epoch

	};

} // end namespace mnistnn

#endif // DATA_LOADER_H
