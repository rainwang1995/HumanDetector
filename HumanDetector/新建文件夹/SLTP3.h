#ifndef SLTP3_H
#define SLTP3_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"
#include "Utils.h"
//using namespace cv;
using namespace std;
class SLTP3 :public DetectionAlgorithm
{
public:
	SLTP3()
	{
		setDefaultParams();
		cal_params();
	}

	SLTP3(cv::Size _winSize, cv::Size _cellSize, int _signThreshold = 30)
		:winSize(_winSize), cellSize(_cellSize), signThreshold(_signThreshold)
	{
		cal_params();
	}
	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	virtual int getFeatureLen() const { return featurelen; }
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	//virtual void setSvmDetector(const string& xmlfile);//set from file
	virtual void loadSvmDetector(const string& xmlfile);
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;
	//virtual void set_signThreshold(const int _signThreshold) { signThreshold = _signThreshold; }

	virtual ~SLTP3() { maskdx.release(), maskdy.release(); sltpsvm.release(); }
private:
	void setDefaultParams();
	void cal_params();

	void compute_dxdy(const cv::Mat& img, cv::Mat& dx, cv::Mat& dy)const;
	void compute_sign(const cv::Mat& dimg, cv::Mat& signimg,int thrpos,int thrneg)const;
	
	void compute_histcell(const cv::Mat& modelmap, float* hist)const;
	void compute_histimg(const cv::Mat& modemap, vector<float>& features)const;
	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	void get_winfeature(vector<float>& featuresimg, vector<float>& featureswin, cv::Point& startpos, int cellimgcols)const;
	void compute_thr(const cv::Mat& imgd, cv::Mat& thrpos, cv::Mat& thrneg)const;
	void compute_signimg(const cv::Mat& imgd, cv::Mat& signd)const;

	void compute_modemap(const cv::Mat& img, cv::Mat& modemap)const;
public:
	cv::Size winSize;//检测窗口大小
	cv::Size cellSize;//分块大小
private:
	int signThreshold;
	int numCellR;//每行cell数量
	int numCellC;//每列cell数量
	int numCellPerWin;
	cv::Mat maskdx;
	cv::Mat maskdy;
	int featurelen;
	int signarry[3][3];
	//int nlevels;
	//float scale0;
	cv::Ptr<cv::ml::SVM> sltpsvm;
	//vector<float> svmvec;
	//double rho;
};

#endif
