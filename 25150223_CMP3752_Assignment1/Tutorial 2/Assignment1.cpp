// Include headers required for image processing
#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

int main(int argc, char** argv) {
	// ---------------- Handle Command Line Options ----------------
	// Variables defined to handle command line inputs
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "";
	string image_number;
	int bin_size;
	string bin_size_number;
	
	// Ask the user which image they want to be processed. 1 & 2 are greyscale images, 3 & 4 are colour images
	cout << "Which image would you like to be processed?:" << endl;
	cout << "	1 - test.pgm" << endl;
	cout << "	2 - testLarge.pgm" << endl;
	cout << "	3 - testColour.pgm" << endl;
	cout << "	4 - testColourLarge.pgm" << endl;

	// Assign choice to 'image_filename'
	while (image_filename == "") {
		cout << ">> ";
		getline(cin, image_number); // 'getline' takes string only, checks that no whitespace is inputted
		if (image_number == "1") {
			image_filename = "test.pgm";
		}
		else if (image_number == "2") {
			image_filename = "testLarge.pgm";
		}
		else if (image_number == "3") {
			image_filename = "testColour.pgm";
		}
		else if (image_number == "4") {
			image_filename = "testColourLarge.pgm";
		}
		else {
			cout << "Please enter a number as stated above" << endl;
		}
	}

	// Ask user which bin count they want, out of the factors of 256
	cout << "How many bins would you like the histogram to have?:" << endl;
	cout << "	1 - 1 bins" << endl;
	cout << "	2 - 2 bins" << endl;
	cout << "	3 - 4 bins" << endl;
	cout << "	4 - 8 bins" << endl;
	cout << "	5 - 16 bins" << endl;
	cout << "	6 - 32 bins" << endl;
	cout << "	7 - 64 bins" << endl;
	cout << "	8 - 128 bins" << endl;
	cout << "	9 - 256 bins" << endl;

	// Assign choice to 'bin_size'
	while (true) {
		cout << ">> ";
		getline(cin, bin_size_number);
		if (bin_size_number == "1") {
			bin_size = 1;
			break;
		}
		else if (bin_size_number == "2") {
			bin_size = 2;
			break;
		}
		else if (bin_size_number == "3") {
			bin_size = 4;
			break;
		}
		else if (bin_size_number == "4") {
			bin_size = 8;
			break;
		}
		else if (bin_size_number == "5") {
			bin_size = 16;
			break;
		}
		else if (bin_size_number == "6") {
			bin_size = 32;
			break;
		}
		else if (bin_size_number == "7") {
			bin_size = 64;
			break;
		}
		else if (bin_size_number == "8") {
			bin_size = 128;
			break;
		}
		else if (bin_size_number == "9") {
			bin_size = 256;
			break;
		}
		else {
			cout << "Please enter a number as stated above" << endl;
		}
	}
	int min_intesity = 0;
	int max_intensity = 256;

	double bin_width = (double)(max_intensity - min_intesity) / (double)(bin_size);

	cimg::exception_mode(0); // Sets the exception mode of the library to 0, disabling them

	// Try-catch block for error handling
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input"); // Display the input image to be processed

		// A 3x3 convolution mask implementing an averaging filter
		vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };

		// ---------------- Host operations ----------------
		cl::Context context = GetContext(platform_id, device_id); // Select computing devices
		cout << "Running on: " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl; // Display the current device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); // A queue where commands will be pushed for the device
		cl::Program::Sources sources; // Load & build the device code
		AddSources(sources, "kernels/my_kernels.cl"); // Adds 'my_kernels.cl' to the list of sources to be compiled and linked to the OpenCL program
		cl::Program program(context, sources); // Creates an OpenCL program from the code in 'sources'

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}

		// ---------------- Memory Allocation ----------------
		// Host - input
		typedef int mytype;
		std::vector<mytype> H(bin_size); // Number of bins defined by user
		size_t histsize = H.size() * sizeof(mytype);

		// Device - buffers, to store data to be used by kernels
		cl::Buffer _image_input(context, CL_MEM_READ_ONLY, image_input.size()); // Will only be read by the OpenCL kernel
		cl::Buffer _hist_output(context, CL_MEM_READ_WRITE, histsize);				// Can be read from and written to by the OpenCL kernel...
		// cl::Buffer _hist_output_B(context, CL_MEM_READ_WRITE, histsize); // ATTEMPT
		cl::Buffer _cumu_hist_output(context, CL_MEM_READ_WRITE, histsize);	// <<...
		cl::Buffer _LUT_output(context, CL_MEM_READ_WRITE, histsize);					// <<...
		cl::Buffer _image_output(context, CL_MEM_READ_WRITE, image_input.size());		// <<

		// ---------------- Device Operations ----------------
		// Copy images to device memory
		queue.enqueueWriteBuffer(_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(_hist_output, 0, 0, histsize);

		// ---------------- Setup and execute the kernels ----------------
		// 1st kernel plots a histogram of the frequency of each pixel value in the image
		cl::Kernel kernel_hist = cl::Kernel(program, "hist");
		kernel_hist.setArg(0, _image_input);
		kernel_hist.setArg(1, _hist_output);
		kernel_hist.setArg(2, bin_width);
		cl::Event hist_event; // Declares an OpenCL event

		// Adds following operations to queues to execute on the device
		queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &hist_event); // Executed on a 1D range
		queue.enqueueReadBuffer(_hist_output, CL_TRUE, 0, histsize, &H[0]);
		// vector<mytype> B(bin_size); // ATTEMPT
		vector<mytype> CH(bin_size);
		// queue.enqueueFillBuffer(_hist_output_B, 0, 0, histsize); // ATTEMPT
		queue.enqueueFillBuffer(_cumu_hist_output, 0, 0, histsize);
		
		// 2nd kernel call plots a Cumulative Histogram of the total pixels in the image
		cl::Kernel kernel_cumu_hist = cl::Kernel(program, "cumu_hist"); // Could be "scan_hs" if it worked...
		kernel_cumu_hist.setArg(0, _hist_output);
		// kernel_cumu_hist.setArg(1, _hist_output_B); // ATTEMPT
		kernel_cumu_hist.setArg(1, _cumu_hist_output); // Should be '2' if "scan_hs" worked
		cl::Event cumu_hist_event;
		queue.enqueueNDRangeKernel(kernel_cumu_hist, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &cumu_hist_event);
		queue.enqueueReadBuffer(_cumu_hist_output, CL_TRUE, 0, histsize, &CH[0]);
		vector<mytype> LUT(bin_size);
		queue.enqueueFillBuffer(_LUT_output, 0, 0, histsize);

		// 3rd kernel call creates a Lookup Table of the new pixel values
		cl::Kernel kernel_LUT = cl::Kernel(program, "LUT");
		kernel_LUT.setArg(0, _cumu_hist_output);
		kernel_LUT.setArg(1, _LUT_output);
		kernel_LUT.setArg(2, bin_width);
		cl::Event LUT_event;
		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &LUT_event);
		queue.enqueueReadBuffer(_LUT_output, CL_TRUE, 0, histsize, &LUT[0]);

		// The final kernel assigns the new pixel values from the Lookup Table to the output image with a higher contrats to the original input image
		cl::Kernel kernel_e_output = cl::Kernel(program, "e_output");
		kernel_e_output.setArg(0, _image_input);
		kernel_e_output.setArg(1, _LUT_output);
		kernel_e_output.setArg(2, _image_output);
		kernel_e_output.setArg(3, bin_width);
		cl::Event e_output_event;
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueNDRangeKernel(kernel_e_output, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &e_output_event);
		queue.enqueueReadBuffer(_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		// Output the Histogram and its execution time and memory transfer in nanoseconds
		cout << endl;
		cout << "Histogram: " << H << endl;
		cout << "Histogram kernel execution time: " << hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl; // Print kernel execution time in nanoseconds
		cout << GetFullProfilingInfo(hist_event, ProfilingResolution::PROF_NS) << endl; // Prints profiling information of the OpenCL event in nanoseconds
		cout << endl;

		// Output the Cumulative Histogram and its execution time in nanoseconds
		cout << "Cumulative Histogram: " << CH << endl;
		cout << "Cumulative Histogram kernel execution time: " << cumu_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumu_hist_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl;
		cout << GetFullProfilingInfo(cumu_hist_event, ProfilingResolution::PROF_NS) << endl;
		cout << endl;

		// Output the Lookup Table and its execution time in nanoseconds
		cout << "LUT: " << LUT << endl;
		cout << "LUT kernel execution time: " << LUT_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - LUT_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" <<endl;
		cout << GetFullProfilingInfo(LUT_event, ProfilingResolution::PROF_NS) << endl;
		cout << endl;

		// Output the execution time of the e_output (4th) kernel in nanoseconds
		cout << "Output processing kernel execution time: " << e_output_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - e_output_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl;
		cout << GetFullProfilingInfo(e_output_event, ProfilingResolution::PROF_NS) << endl;
		cout << endl;

		// Output total execution time of the program from when 'hist_event' is enqueued by the host until 'e_output_event' has finished execution
		cout << "Total program execution time: " << e_output_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - hist_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() << " [ns]" << endl;

		// Display the processed output image
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		// A loop to check that neither the input or output images have been closed before continuing execution of the program
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	// Catch end of the try-catch block, cathces errors and outputs message to the user
	catch (const cl::Error& err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}
	catch (CImgException& err) {
		cerr << "ERROR: " << err.what() << endl;
	}

	return 0;
}
