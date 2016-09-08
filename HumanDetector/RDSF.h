#ifndef RDSF_H
#define RDSF_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"

//using namespace cv;
using namespace std;
class RDSF :public DetectionAlgorithm
{
public:
	RDSF()
	{
		setDefaultParams();
		cal_params();
	}

	RDSF(cv::Size _winSize, cv::Size _cellSize, int _nbins = 30)
		:winSize(_winSize), cellSize(_cellSize), n_bins(_nbins)
	{
		cal_params();
	}
	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	virtual int getFeatureLen() const { return featurelen; }
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	//virtual void setSvmDetector(const string& xmlfile);//set from file
	virtual void loadSvmDetector(const string& xmlfile);
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights,
		double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold = 0, 
		cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, 
		double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual ~RDSF() { rdsfsvm.release(); }
private:
	void setDefaultParams();
	void cal_params();

	void compute_binmap(const cv::Mat& img, cv::Mat& binmap)const;
	void compute_intermap(const cv::Mat& binmap, vector<cv::Mat>& intergmap)const;//计算积分图像
	//void compute_feature(const vector<cv::Mat>& integralmaps, vector<float>& features)const;
	int  compute_histvalue(const cv::Mat& integralmap, cv::Rect pos)const;
	void compute_hists(const vector<cv::Mat>& integralmaps, cv::Rect pos, vector<float>& hist)const;
	void normalise_hist(vector<float>& hist)const;
	float compute_similarity(const vector<float>& hista, const vector<float>& histb)const;
	void compute_win(const vector<cv::Mat>& integralmap, cv::Rect roi, vector<float>& features)const;

	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
public:
	cv::Size winSize;//检测窗口大小
	cv::Size cellSize;//分块大小
private:
	int featurelen;
	int n_bins;

	cv::Ptr<cv::ml::SVM> rdsfsvm;

};

#endif

