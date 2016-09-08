#ifndef MPLDPK_H
#define MPLDPK_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"

using namespace std;

class MPLDPK :public DetectionAlgorithm
{
public:
	MPLDPK() :featurelenblock(56) { setDefaultParams(); cal_params(); }
	MPLDPK(cv::Size _winSize, int _plevels = 2, int _k = 3) :featurelenblock(56), winSize(_winSize), K(_k), plevels(_plevels)
	{
		cal_params();
	}
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	virtual void loadSvmDetector(const string& xmlfile);
	virtual int getFeatureLen()const { return featurelen; }

	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold = 0,
		cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights,
		double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(),
		double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;

	virtual ~MPLDPK() { ldpksvm.release(); masks.clear(); };
public:
	cv::Size winSize;
	int K;
private:
	cv::Size blockSize;
	int blockrows;
	int blockcols;
	int blocksinlevels;
	int featurelen;
	int plevels;
	const int featurelenblock;
	vector<cv::Mat> masks;
	vector<cv::Rect> parts;
	cv::Mat lookUpTable;
private:
	cv::Ptr<cv::ml::SVM> ldpksvm;
private:
	void setDefaultParams();
	void cal_params();

	void compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimg)const;
	int  compute_histvalue(const cv::Mat& integralmap, cv::Rect pos)const;
	void compute_hists(const vector<cv::Mat>& integralmaps, cv::Rect pos, float* hist)const;
	void normalise_hist(float* blockhist)const;
	void compute_integralmap(const cv::Mat& ltpimg, vector<cv::Mat>& integralmaps)const;

	//void compute_feature(const vector<cv::Mat>& integralmaps, vector<float>& features)const;

	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
public:
	void compute_integralmaps(const cv::Mat& src, vector<cv::Mat>& integralmaps)const;
	void compute_histwin(const vector<cv::Mat>& integralmap, cv::Rect roi, vector<float>& features)const;

};
#endif

