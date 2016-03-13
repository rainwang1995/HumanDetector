#ifndef HDDDETECT_H
#define HDDDETECT_H
#include "opencv2/core/core.hpp"
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"

class HDD :public DetectionAlgorithm
{
public:
	enum { L2Hys = 0 };
	enum { DEFAULT_NLEVELS = 64 };
	HDD() :winSize(64, 128), blockSize(16, 16), blockStride(8, 8),
		cellSize(8, 8), nbins(9), derivAperture(1), winSigma(-1), histogramNormType(HDD::L2Hys),
		L2HysThreshold(0.2)
	{
		cal_parms();
	}

	HDD(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride,
		cv::Size _cellSize, int _nbins = 9, int _derivAperture = 1, double _winSigma = -1,
		int _histogramNormType = HDD::L2Hys, double _L2HysThreshold = 0.2)
		:winSize(_winSize), blockSize(_blockStride),
		blockStride(_blockStride), cellSize(_cellSize), nbins(_nbins), derivAperture(_derivAperture),
		winSigma(_winSigma), histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold)

	{
		cal_parms();
	}


	HDD(const HDD& d)
	{
		d.copyTo(*this);
	}

	virtual int getFeatureLen()const { return featurenlen; }
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	virtual void loadSvmDetector(const string& xmlfile);
	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations,
		vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;

	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold = 0,
		cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;

	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(),
		const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(),
		double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;


	double getWinSigma() const;
	void copyTo(HDD& c) const;
	void computeGradient(const cv::Mat& img, cv::Mat& grad, cv::Mat& angleOfs,
		cv::Size paddingTL = cv::Size(), cv::Size paddingBR = cv::Size()) const;

	virtual ~HDD() {}
private:
	void cal_parms();
	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	
public:
	cv::Size winSize;
	double L2HysThreshold;
	cv::Size blockSize;
	cv::Size blockStride;
	cv::Size cellSize;
	int nbins;
private:
	int derivAperture;
	double winSigma;
	int histogramNormType;
	//CV_PROP bool gammaCorrection;
	//vector<float> svmDetector;
	cv::Ptr<cv::ml::SVM> hddsvm;

	int featurenlen;
};
#endif
