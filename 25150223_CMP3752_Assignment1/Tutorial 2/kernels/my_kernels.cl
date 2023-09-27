// Histogram kernel
kernel void hist(global const uchar* A, global int* H, double bin_width) {
	int id = get_global_id(0);

	int bin_index = (int)((double)(A[id]) / bin_width); // Take value as a bin index, assumes 'H' has been initialised to 0.
	atomic_inc(&H[bin_index]); // Increments the bin in a thread-safe manner, may lead to performance... 
							   // ... issues for large datasets due to the serialisation of memory access.
}

// Cumulative Histogram SCAN kernel, based on atomic operations
kernel void cumu_hist(global int* H, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++) // Loops over all elements in the 'H' buffer and adds the value at index...
		atomic_add(&CH[i], H[id]);	 // ... 'id' to all elements at indices > id in the 'CH' buffer.
}

/*
// Hillis-Steele basic inclusive scan
// Requires additional buffer 'B' to avoid data overwrite 
kernel void scan_hs(global int* H, global int* B, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = H[id];
		if (id >= stride)
			B[id] += H[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		CH = H; H = B; B = CH; // swap A & B between steps
	}
}
*/

// Lookup Table (LUT) kernel, maps pixel values from 'CH' to result in an equalised histogram.
kernel void LUT(global int* CH, global int* LUT, double bin_width) {
	int id = get_global_id(0);
	LUT[id] = CH[id] * 255 / CH[(int)((double)255 / bin_width)]; //  Normalises the cumulative histogram value and scaling it to the range 0-255.
}

// Applies the LUT to each pixel value in 'A' and writes the result to the corresponding location in 'B' (the output image).
kernel void e_output(global uchar* A, global int* LUT, global uchar* B, double bin_width) {
	int id = get_global_id(0);
	B[id] = LUT[(int)((double)A[id] / bin_width)];
}