#ifndef HONV_H
#define HONV_H

#include "opencvHeader.h"
#include <string>
#include <vector>
#include <armadillo>
using namespace std;

const double PI = arma::datum::pi;
const int MAX_DIFF_16 = 1500;
const double PROPORTION_OF_DIVIDING_PHI_BINS = 0.7;

class HONV
{
	struct spatialWeights
	{
		arma::Mat<float> x1y1;
		arma::Mat<float> x2y1;
		arma::Mat<float> x1y2;
		arma::Mat<float> x2y2;

		arma::ivec cellX;
		arma::ivec cellY;

		spatialWeights(cv::Size size = cv::Size(16, 16))
		{
			x1y1.resize(size.height, size.width);
			x2y1.resize(size.height, size.width);
			x1y2.resize(size.height, size.width);
			x2y2.resize(size.height, size.width);

			cellX.resize(size.width);
			cellY.resize(size.height);
		}
	};

	struct trilinearWeights
	{
		arma::fmat x1_y1_z1;
		arma::fmat x1_y1_z2;
		arma::fmat x2_y1_z1;
		arma::fmat x2_y1_z2;
		arma::fmat x1_y2_z1;
		arma::fmat x1_y2_z2;
		arma::fmat x2_y2_z1;
		arma::fmat x2_y2_z2;
		trilinearWeights(cv::Size size = cv::Size(16, 16))
		{
			x1_y1_z1.resize(size.height, size.width);
			x1_y1_z2.resize(size.height, size.width);
			x2_y1_z1.resize(size.height, size.width);
			x2_y1_z2.resize(size.height, size.width);
			x1_y2_z1.resize(size.height, size.width);
			x1_y2_z2.resize(size.height, size.width);
			x2_y2_z1.resize(size.height, size.width);
			x2_y2_z2.resize(size.height, size.width);
		}
	};
public:
	HONV() :winSize(64, 128), blockSize(16,16),cellSize(8, 8), bin_numtheta(4),bin_numphi(9), HONV_difference(1),overlap(0,0),spatialweights(blockSize)
	{
		cal_para();
	}
	HONV(cv::Size _winSize,cv::Size _blockSize,cv::Size _cellSize,int _HONV_diff=1,int _bin_numtheta =4,int _bin_numphi=9,cv::Size _overlap=cv::Size(0,0)):
		winSize(_winSize),blockSize(_blockSize),cellSize(_cellSize),HONV_difference(_HONV_diff), bin_numtheta(_bin_numtheta),bin_numphi(bin_numphi),overlap(_overlap), spatialweights(blockSize)
	{
		cal_para();
	}

	int getFeatureLen()const { return featureLen; }
	void compute(cv::Mat& img, vector<float>& feature);

	virtual ~HONV(){}
private:
	//HONV_cell cell;
	//HONVWindow DetWin;
	cv::Size winSize;
	cv::Size cellSize;
	cv::Size blockSize;
	cv::Size overlap;
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

	spatialWeights spatialweights;
	//trilinearWeights trilinearweights;

	cv::Size blockStride;

	//int blockStride;

	int bin_numtheta;
	int bin_numphi;
	cv::Mat maskdx;
	cv::Mat maskdy;
	arma::Mat<float> gaussianweights;
	//arma::Mat<float> spaitalweights;

	
	//cv::Mat_<float> weights;
private:
	void cal_para();
	void compute_dxdy(cv::Mat& src,cv::Mat& dximg,cv::Mat& dyimg);
	void compute_theta_phi(cv::Mat& dximg, cv::Mat& dyimg, arma::Mat<float>& theta, arma::Mat<float>& phi);
	void compute_hist(arma::Mat<float>& theta, arma::Mat<float>& phi, vector<float>& thetahist, vector<float>& phihist);
	void compute_hist(arma::fmat& theta, arma::fmat& phi, vector<float>& hist);

	void compute_wight();

	void compute_lowerHistBin(arma::fmat& x, float binwidth, arma::fmat& bincenter, arma::imat& binindex);
	void compute_lowerHistBin(arma::fvec& x, float binwidth, arma::fvec& bincenter, arma::ivec& binindex);
	void compute_HistBin(arma::fmat& x, float binwidth, arma::fmat& bincenter, arma::imat& binindex);
	void compute_lowerHistBin(arma::fmat& theta, arma::fmat& phi,arma::fcube& bincenter,arma::icube& binindex);


	void compute_trilinear(arma::fmat& wz, spatialWeights& weights,trilinearWeights triweights);
};



#endif // !HONV_H

