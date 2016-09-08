#ifndef PSLTP_H
#define PSLTP_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"

using namespace std;

class PSLTP :public DetectionAlgorithm
{
public:

	virtual void loadSvmDetector(const string& xmlfile);
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);

private:
	cv::Ptr<cv::ml::SVM> psltpsvm;
	cv::Mat maskdx, maskdy;
	int signThreshold;
private:
	void compute_dxdy(const cv::Mat& img, cv::Mat& dx, cv::Mat& dy)const;
	void compute_sign(const cv::Mat& dimg, cv::Mat& signimg)const;

	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
};
#endif
