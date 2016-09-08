#ifndef LTDP2_h
#define LTDP2_h
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"
//using namespace cv;
using namespace std; 

class LTDP2 :public DetectionAlgorithm
{
public:
	LTDP2():featurelenblock(118)
	{
		setDefaultParams();
		cal_params();
	}
	LTDP2(cv::Size _winSize, cv::Size _blockSize,int _pthreshold=120) 
		:winSize(_winSize), blockSize(blockSize),pthreshold(_pthreshold),featurelenblock(118)
	{
		cal_params();
	}

	virtual int getFeatureLen()const { return featurelen; }
	virtual void compute(const cv::Mat& img, vector<float>& features)const;

	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	virtual void loadSvmDetector(const string& xmlfile);
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, 
		vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;

	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold = 0, 
		cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;

	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), 
		const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), 
		double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;

	virtual void set_signThreshold(const int _pthreshold) { pthreshold = _pthreshold; }

	virtual ~LTDP2() { masks.clear(); ltdpsvm.release(); svmvec.clear(); }
public:
	cv::Size blockSize;
	cv::Size winSize;

private:
	int numBlockR;
	int numBlockC;
	int featurelen;
	int numBlockPerWin;
	const int featurelenblock;
	vector<cv::Mat> masks;
	int pthreshold;

	cv::Mat lookUpTable;
	cv::Ptr<cv::ml::SVM> ltdpsvm;
	vector<float> svmvec;
	double rho;
private:
	void setDefaultParams();
	void cal_params();

	void compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimgpos,cv::Mat& ltpimgneg)const;
	void compute_histblock(const cv::Mat& ltppos, const cv::Mat& ltpneg, float* feature)const;
	void compute_histwin(const cv::Mat& ltppos, const cv::Mat& ltpneg, vector<float>& features)const;
	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	void compute_Ltpimg(const cv::Mat& src, cv::Mat& uniformp, cv::Mat& uniformn)const;
};

#endif
