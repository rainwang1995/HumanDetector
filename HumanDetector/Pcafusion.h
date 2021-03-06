#ifndef PCAFUSION_H
#define PCAFUSION_H

#include "opencvHeader.h"
#include <iostream>
#include <vector>
#include <string>
#include "Algorithm.hpp"
#include "HDD.h"
#include "PLDPK.h"
#include <armadillo>

using namespace std;

class PCAFUSION :public DetectionAlgorithm
{
public:

	PCAFUSION() :winSize(64, 128)
	{

	}

	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	virtual void loadSvmDetector(const string& xmlfile);

	virtual void loadccamatrix(const string& ymlpca);
	virtual int getFeatureLen() const;
	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;

	virtual ~PCAFUSION() {}

public:
	cv::Size winSize;

private:
	int featurelen;
	HDD hdddetector;
	PLDPK pldpkdetector;
	cv::Ptr<cv::ml::SVM> ccasvm;
	cv::Mat wxcca, wycca;
	arma::fmat fwxcca, fwycca;
private:
	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	void fusionfeature(vector<float>& f1, vector<float>& f2, vector<float>& f3)const;
};
#endif
