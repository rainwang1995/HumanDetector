#ifndef HONVNW_H
#define HONVNW_H

#include "opencvHeader.h"
#include <string>
#include <vector>
#include <armadillo>
#include <opencv2/ml.hpp>

using namespace std;

const double PI = arma::datum::pi;
const int MAX_DIFF_16 = 1500;
const double PROPORTION_OF_DIVIDING_PHI_BINS = 0.7;
const float NORMTHR = 0.4;

class HONVNW
{
public:
	HONVNW() :winSize(64, 128), blockSize(16, 16), cellSize(8, 8), bin_numtheta(4), bin_numphi(9), HONV_difference(1),nlevels(64)
	{
		cal_para();
	}
	HONVNW(cv::Size _winSize, cv::Size _blockSize, cv::Size _cellSize, int _HONV_diff = 1, int _bin_numtheta = 4, int _bin_numphi = 9,int _nlevels=64) :
		winSize(_winSize), blockSize(_blockSize), cellSize(_cellSize), HONV_difference(_HONV_diff), bin_numtheta(_bin_numtheta), bin_numphi(bin_numphi),nlevels(_nlevels)
	{
		cal_para();
	}

	int getFeatureLen()const { return featureLen; }
	void compute(cv::Mat& img, vector<float>& feature)const;
	void setSvmDetector(cv::Ptr<cv::ml::SVM>& _svm);
	void setSvmDetector(const string& xmlfile);
	void detect(cv::Mat& img, vector<cv::Point>& foundLocations, vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(),const vector<cv::Point>& locations=vector<cv::Point>()) const;
	void detectMultiScale(cv::Mat& img, vector<cv::Rect>& foundloacations, vector<double>& weights, double hitThreshold = 0, cv::Size winStride = cv::Size(), double scale = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;

	virtual ~HONVNW() { maskdx.release(); maskdy.release(); svm.release(); }
public:
	//HONV_cell cell;
	//HONVWindow DetWin;
	cv::Size winSize;
	cv::Size cellSize;
	cv::Size blockSize;
	cv::Size overlap;
	int nlevels;

private:
	int HONV_difference;
	int featureLen;
	int cellnumR;
	int cellnumC;
	int cellnumRblock;
	int cellnumCblock;
	int blocknumR;
	int blocknumC;
	int numCellsPerBlock;
	int numBlocksPerWin;

	cv::Size blockStride;

	int bin_numtheta;
	int bin_numphi;
	cv::Mat maskdx;
	cv::Mat maskdy;

	cv::Ptr<cv::ml::SVM> svm;

private:
	void cal_para();
	void compute_dxdy(cv::Mat& src, cv::Mat& dximg, cv::Mat& dyimg)const;
	void compute_theta_phi(cv::Mat& dximg, cv::Mat& dyimg, arma::Mat<float>& theta, arma::Mat<float>& phi)const;

	void compute_HistBin(arma::fmat& theta,arma::fmat& phi,arma::fcube& bincenter, arma::icube& binindex)const;
	void compute_HistBin(arma::fmat& x, float binwidth, arma::fmat& bincenter, arma::imat& binindex)const;

	void compute_hist_cell(arma::icube& binindex, arma::ivec& cellhist)const;
	void compute_hist_block(arma::icube& binindex, arma::fvec& blockhist)const;

	void compute_hist(arma::icube& binindex, arma::fvec& hist)const;

	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
};



#endif // !HONV_H
