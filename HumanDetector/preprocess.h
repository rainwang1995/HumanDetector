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
	static void pixelFilter(cv::Mat& src,cv::Mat& dst);
	static void contextFilter(cv::Mat& src, cv::Mat& dst);
private:
	static deque<cv::Mat> framequeue;
};
#endif
