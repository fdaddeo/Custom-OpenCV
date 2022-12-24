// OpenCV
#include "Template/CustomCV.h"

// std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList
{
	std::string image_name; //!< image file name
	int wait_t;				//!< waiting time
};

bool ParseInputs(ArgumentList &args, int argc, char **argv);

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;
	int stride = 1;

	//////////////////////
	// parse argument list:
	//////////////////////
	ArgumentList args;
	if (!ParseInputs(args, argc, argv))
	{
		exit(0);
	}

	while (!exit_loop)
	{
		// generating file name
		//
		// multi frame case
		if (args.image_name.find('%') != std::string::npos)
			sprintf(frame_name, (const char *)(args.image_name.c_str()), frame_number);
		else // single frame case
			sprintf(frame_name, "%s", args.image_name.c_str());

		// opening file
		std::cout << "Opening " << frame_name << std::endl;

		cv::Mat image = cv::imread(frame_name);
		if (image.empty())
		{
			std::cout << "Unable to open " << frame_name << std::endl;
			return 1;
		}

		// display image
		cv::namedWindow("original image", cv::WINDOW_NORMAL);
		cv::imshow("original image", image);

		// PROCESSING

		// convert to grey scale for following processings
		cv::Mat grey;

		cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
		cv::namedWindow("grey", cv::WINDOW_NORMAL);
		cv::imshow("grey", grey);

		cv::Mat customBlurred;
		cv::Mat bilateralFilter;
		cv::Mat customBilateralFilter;
		cv::Mat magnitude;
		cv::Mat orientation;
		cv::Mat maximaSuppressed;
		cv::Mat canny;
		cv::Mat cannyOutput;
		cv::Mat houghOutput;

		//// Bilateral Filter
		custom_cv::bilateralFiltering(grey, customBilateralFilter, 3, 2.0f, 5.0f);
		////

		//// Canny
		custom_cv::gaussianBlur(grey, 1.0f, 1, customBlurred, stride);
		customBlurred.convertTo(customBlurred, CV_8UC1);
		custom_cv::sobel3x3(customBlurred, magnitude, orientation);
		custom_cv::findPeaks(magnitude, orientation, maximaSuppressed);
		custom_cv::doubleTh(maximaSuppressed, cannyOutput, 50, 150);
		////

		//// Hough
		std::vector<cv::Vec2f> lines;
		custom_cv::houghTransform(cannyOutput, lines, 180);
		custom_cv::drawLinesOnImage(image, houghOutput, lines);
		////

		customBilateralFilter.convertTo(customBilateralFilter, CV_8UC1);
		cv::namedWindow("Bilateral Filter", cv::WINDOW_NORMAL);
		cv::imshow("Bilateral Filter", customBilateralFilter);

		cv::namedWindow("Canny", cv::WINDOW_NORMAL);
		cv::imshow("Canny", cannyOutput);

		cv::namedWindow("Hough", cv::WINDOW_NORMAL);
		cv::imshow("Hough", houghOutput);

		// wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout << "key " << int(key) << std::endl;

		// here you can implement some looping logic using key value:
		//  - pause
		//  - stop
		//  - step back
		//  - step forward
		//  - loop on the same frame

		switch (key)
		{
		case 's':
			if (stride != 1)
				--stride;
			std::cout << "Stride: " << stride << std::endl;
			break;
		case 'S':
			++stride;
			std::cout << "Stride: " << stride << std::endl;
			break;

		case 'c':
			cv::destroyAllWindows();
			break;
		case 'p':
			std::cout << "Mat = " << std::endl
					  << image << std::endl;
			break;
		case 'g':
			break;
		case 'q':
			exit(0);
			break;
		}

		frame_number++;
	}

	return 0;
}

#if 0
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|in.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0     |wait before next frame (ms)}"
      "{help    h|<none>|produce help message}"
      );

  if(parser.has("help"))
  {
    parser.printMessage();
    return false;
  }

  args.image_name = parser.get<std::string>("input");
  args.wait_t     = parser.get<int>("wait");

  return true;
}
#else

#include <unistd.h>
bool ParseInputs(ArgumentList &args, int argc, char **argv)
{
	int c;
	args.wait_t = 0;

	while ((c = getopt(argc, argv, "hi:t:")) != -1)
		switch (c)
		{
		case 't':
			args.wait_t = atoi(optarg);
			break;
		case 'i':
			args.image_name = optarg;
			break;
		case 'h':
		default:
			std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
			std::cout << "exit:  type q" << std::endl
					  << std::endl;
			std::cout << "Allowed options:" << std::endl
					  << "   -h                       produce help message" << std::endl
					  << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
					  << "   -t arg                   wait before next frame (ms)" << std::endl
					  << std::endl;
			return false;
		}
	return true;
}

#endif
