#ifndef SLTP_H
#define SLTP_H
#include "opencvHeader.h"
#include <armadillo>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>

//using namespace cv;
using namespace std;
class SLTP
{
public:
	SLTP():winSize(64,128),cellSize(8,8)
	{
		setDefaultParams();
		cal_params();
	}

	SLTP(cv::Size _winSize, cv::Size _cellSize, int _signThreshold = 30)
		:winSize(_winSize),cellSize(_cellSize),signThreshold(_signThreshold)
	{
		cal_params();
	}

	void compute(const cv::Mat& img, vector<float>& features)const;
	int getFeaturelen() const{ return featurelen; }
	void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	void setSvmDetector(const string& xmlfile);//set from file

	void detect(const cv::Mat& img,vector<cv::Point>& foundlocations,cv::Size winStride=cv::Size(),vector<cv::Point>& locations=vector<cv::Point>())const;
	void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, cv::Size winStride = cv::Size(), double nlevels=64,double scale0 = 1.05, double hitThreshold = 0, double finalThreshold = 2.0, bool usemeanshift = false)const;
private:
	void setDefaultParams();
	void cal_params();

	void compute_dxdy(const cv::Mat& img, cv::Mat& dx, cv::Mat& dy)const;
	void compute_sign(const cv::Mat& dimg, cv::Mat& signimg)const;
	void compute_histcell(const cv::Mat& signimgx, const cv::Mat& signimgy, vector<float>& hist)const;
	void compute_histcell(const cv::Mat& signimgx, const cv::Mat& signimgy, float* hist)const;
	void compute_histwin(const cv::Mat& signimgx, const cv::Mat& signimgy, vector<float>& hist)const;
	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	
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
};

#endif
