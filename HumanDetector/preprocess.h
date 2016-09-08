#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <iostream>
#include "opencvHeader.h"
#include <deque>
//#include "fmm.hpp"
using namespace std;

const int innerBandThreshold = 2;
const int outerBandThreshold = 5;

class preProcessing
{
public:
	static void pixelFilter(cv::Mat& src,cv::Mat& dst);
	//static void nearesetFilter(cv::Mat& src, cv::Mat& dst);
	static void contextFilter(cv::Mat& src, cv::Mat& dst);
	static void inpaintrawdepth(cv::Mat& src, cv::Mat& dst,double inpaintRange,float alpha);
private:
	static deque<cv::Mat> framequeue;
};

void filterfunc1(cv::Mat& src, cv::Mat& dst);
void filterfunc2(cv::Mat& src, cv::Mat& dst);
void filterfunc3(cv::Mat& src, cv::Mat& dst);
void filterfunc4(cv::Mat& src, cv::Mat& dst);
#endif
