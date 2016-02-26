#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <iostream>
#include "opencvHeader.h"
#include <deque>
using namespace std;

const int innerBandThreshold = 2;
const int outerBandThreshold = 5;

class preProcessing
{
public:
	static void pixelFilter(Mat& src,Mat& dst);
	static void contextFilter(Mat& src, Mat& dst);
private:
	static deque<Mat> framequeue;
};
#endif
