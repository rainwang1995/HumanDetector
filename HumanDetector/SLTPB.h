#ifndef SLTPB_H
#define SLTPB_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"

//using namespace cv;
using namespace std;
class SLTPB :public DetectionAlgorithm
{
public:
	SLTPB()
	{
		setDefaultParams();
		cal_params();
	}

	SLTPB(cv::Size _winSize, cv::Size _cellSize,cv::Size _blockSize,cv::Size _blockStride, int _signThreshold = 30)
		:winSize(_winSize), cellSize(_cellSize), signThreshold(_signThreshold)
	{
		cal_params();
	}
	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	virtual int getFeatureLen() const { return featurelen; }
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	//virtual void setSvmDetector(const string& xmlfile);//set from file
	virtual void loadSvmDetector(const string& xmlfile);
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold = 0, 
		cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, 
		double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), 
		double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void set_signThreshold(const int _signThreshold) { signThreshold = _signThreshold; }
	virtual ~SLTPB() { maskdx.release(), maskdy.release(); sltpsvm.release(); }
private:
	void setDefaultParams();
	void cal_params();

	void compute_dxdy(const cv::Mat& img, cv::Mat& dx, cv::Mat& dy)const;
	void compute_sign(const cv::Mat& dimg, cv::Mat& signimg)const;
	void compute_histblock(const cv::Mat& binblock, float* hist)const;
	void compute_histwin(const cv::Mat& win, vector<float>& feature)const;
	void compute_histimage(const cv::Mat& binimg, vector<float> hist)const;
	void compute_binwin(const cv::Mat& signimgx, const cv::Mat& signimgy, cv::Mat& patterns)const;

	void compute_binimg(const cv::Mat& img, cv::Mat& patterns)const;
	void getblockhist(const cv::Mat& blockimg,cv::Point pt, float* blockhist, vector<vector<float> >& imagehist,vector<bool> flags, int blockperrow)const;
	void compute_gaussian();
	void compute_weights();
	void get_winfeature(vector<float>& featuresimg, vector<float>& featureswin, cv::Point& startpos, int blocksperrow)const;

	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	void compute_img(const cv::Mat& img, vector<float>& features)const;//compute a image's SLTP block by block to  accelerate detection speed
	void normalizeBlockHistogram(float* blockhist)const;
public:
	cv::Size winSize;//检测窗口大小
	cv::Size blockSize;
	cv::Size cellSize;//分块大小
	cv::Size blockStride;
private:
	int signThreshold;
	int numCellR;//每行cell数量
	int numCellC;//每列cell数量
	int numBlockR;
	int numBlockC;
	int numBlockWin;
	int numCellPerWin;
	cv::Mat maskdx;
	cv::Mat maskdy;
	int featurelen;
	int signarry[3][3];
	//int nlevels;
	//float scale0;
	cv::Ptr<cv::ml::SVM> sltpsvm;
	cv::Mat_<float> gaussianweights;
	cv::Mat blockweights;
	cv::Mat blockOfs;
	//vector<float> svmvec;
	//double rho;
};

#endif
