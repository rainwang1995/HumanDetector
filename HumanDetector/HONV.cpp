#include "HONV.h"
#include <omp.h>
using namespace cv;

void HONV::compute(Mat & img, vector<float>& feature)
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);
	arma::fmat theta, phi;
	compute_theta_phi(dx, dy, theta, phi);

	vector<float> thetahist, phihist;
	compute_hist(theta, phi, thetahist, phihist);


}
//using namespace arma;
void HONV::cal_para()
{
	cellnumR = winSize.width / cellSize.width;
	cellnumC = winSize.height / cellSize.height;
	int numcellPerWin = cellnumC*cellnumR;

	cellnumRblock = blockSize.width / cellSize.width;
	cellnumCblock = blockSize.height / cellSize.height;

	blockStride.width = blockSize.width-overlap.width;
	blockStride.height = blockSize.height-overlap.height;

	numCellsPerBlock = blockSize.width / cellSize.width*blockSize.height / cellSize.height;

	blocknumR = winSize.width / blockStride.width-overlap.width;
	blocknumC = winSize.height / blockStride.height-overlap.height;
	numBlocksPerWin = blocknumR*blocknumC;

	featureLen = numCellsPerBlock*bin_numphi*bin_numtheta*numBlocksPerWin;

	maskdx.create(1, HONV_difference*2+1, CV_32F);
	maskdy.create(HONV_difference*2+1, 1, CV_32F);

	maskdx.at<float>(0, 0) = -0.5;
	maskdx.at<float>(0, maskdx.cols-1) = 0.5;

	maskdy.at<float>(0, 0) = -0.5;
	maskdy.at<float>(maskdy.rows, 0) = 0.5;

	compute_wight();
}

void HONV::compute_dxdy(cv::Mat & src, cv::Mat & dximg, cv::Mat & dyimg)
{
	filter2D(src, dximg, maskdx.depth(), maskdx);
	filter2D(src, dyimg, maskdy.depth(), maskdy);
}

void HONV::compute_theta_phi(cv::Mat& dximg, cv::Mat& dyimg, arma::Mat<float>& theta, arma::Mat<float>& phi)
{

	arma::Mat<float> dxmat(dximg.ptr<float>(0), dximg.rows, dximg.cols, false);
	arma::Mat<float> dymat(dyimg.ptr<float>(0), dyimg.rows, dyimg.cols, false);

	arma::Mat<float> powmatx = arma::pow(dxmat, 2.0);
	arma::Mat<float> powmaty = arma::pow(dymat, 2.0);

    phi = arma::atan(dymat/dxmat)+PI/2;
	theta = arma::atan(arma::sqrt(powmatx + powmaty))+PI/ 2;
}

void HONV::compute_hist(arma::Mat<float>& theta,arma::Mat<float>&phi, vector<float>& thetahist,vector<float>& phihist)
{
	/*thetahist.resize(cellnumR*cellnumC*bin_numtheta);
	memset(&thetahist[0], 0, thetahist.size()*sizeof(float));*/
	thetahist.reserve(numBlocksPerWin*bin_numtheta*numCellsPerBlock);
	phihist.reserve(numBlocksPerWin*bin_numphi*numCellsPerBlock);

	float binwidththeta = 2 * PI / bin_numtheta;
	float binwidthphi = 2 * PI / bin_numphi;

	//computeLowerHistBin
	arma::fmat bincentertheta, bincenterphi;
	arma::imat binindextheta, binindexphi;

	compute_lowerHistBin(theta, binwidththeta, bincentertheta, binindextheta);
	compute_lowerHistBin(phi, binwidthphi, bincenterphi, binindexphi);

	//双线性插值权重
	arma::Mat<float> weight_ori_theta = 1.0 - (theta - bincentertheta) / binwidththeta;
	arma::Mat<float> weigth_ori_phi = 1.0 - (phi - bincenterphi) / binwidthphi;

	int rindex = 0;
	int cindex = 0;

	arma::Mat<float> gMag(winSize.height, winSize.width,arma::fill::ones);

	trilinearWeights thetaweigths, phiweights;
	arma::fcube blockhisttheta(cellnumCblock + 2, cellnumRblock + 2, bin_numtheta + 2, arma::fill::zeros);
	arma::fcube blockhistphi(cellnumCblock + 2, cellnumRblock + 2, bin_numphi+ 2, arma::fill::zeros);
	arma::fvec hvec(bin_numtheta*numCellsPerBlock);
	arma::fvec pvec(bin_numphi*numCellsPerBlock);

	//scan across all blocks
	for (int j = 0; j < blocknumR;++j)
	{
		for (int i = 0; i < blocknumC;++i)
		{
			blockhisttheta.zeros();
			//theta
			arma::fmat wz1 = weight_ori_theta.submat(rindex, cindex,arma::size(blockSize.width, blockSize.height));
			compute_trilinear(wz1, spatialweights, thetaweigths);

			trilinearWeights f = thetaweigths;

			arma::fmat m = gaussianweights;
			//interpolate magnitudes for binning
			arma::fmat mx1y1z1 = m%f.x1_y1_z1;
			arma::fmat mx1y1z2 = m%f.x1_y1_z2;
			arma::fmat mx2y1z1 = m%f.x2_y1_z1;
			arma::fmat mx2y1z2 = m%f.x2_y1_z2;
			arma::fmat mx1y2z1 = m%f.x1_y2_z1;
			arma::fmat mx1y2z2 = m%f.x1_y2_z2;
			arma::fmat mx2y2z1 = m%f.x2_y2_z1;
			arma::fmat mx2y2z2 = m%f.x2_y2_z2;

			arma::Mat<int> orientationBins = binindextheta.submat(rindex, cindex, arma::size(blockSize.width, blockSize.height));
			arma::Mat<int> orientationBinsphi = binindexphi.submat(rindex, cindex, arma::size(blockSize.width, blockSize.height));

			//initialize block histogram to zero

			for (int x = 0; x < blockSize.width;++x)
			{
				int cx = spatialweights.cellX(x);
				for (int y = 0; y < blockSize.height;++y)
				{
					int cy = spatialweights.cellY(y);
					int z = orientationBins(y, x);

					blockhisttheta(cy, cx, z) = blockhisttheta(cy, cx, z) + mx1y1z1(y, x);
					blockhisttheta(cy, cx, z+1) = blockhisttheta(cy, cx, z+1) + mx1y1z2(y, x);
					blockhisttheta(cy+1, cx, z) = blockhisttheta(cy+1, cx, z) + mx1y2z1(y, x);
					blockhisttheta(cy + 1, cx, z+1) = blockhisttheta(cy + 1, cx, z+1) + mx1y2z2(y, x);
					blockhisttheta(cy, cx+1, z) = blockhisttheta(cy, cx+1, z) + mx2y1z1(y, x);
					blockhisttheta(cy, cx + 1, z+1) = blockhisttheta(cy, cx + 1, z+1) + mx2y1z2(y, x);
					blockhisttheta(cy+1, cx+1, z) = blockhisttheta(cy+1, cx+1, z) + mx2y2z1(y, x);
					blockhisttheta( cy+1, cx+1, z+1) = blockhisttheta(cy+1, cx+1, z+1) + mx2y2z2(y, x);
				}
			}

			//wrap orientation bins
			blockhisttheta.slice(1) = blockhisttheta.slice(1) + blockhisttheta.slice(blockhisttheta.n_slices-1);
			blockhisttheta.slice(blockhisttheta.n_slices-2) = blockhisttheta.slice(blockhisttheta.n_slices - 2) + blockhisttheta.slice(0);

			arma::fcube h = blockhisttheta(arma::span(1, blockhisttheta.n_rows - 2), arma::span(1, blockhisttheta.n_cols - 2), 
				arma::span(1, blockhisttheta.n_slices - 2));

			//converttovec

			for (int i = 0,index=0; i < h.n_rows;++i)
			{
				for (int j = 0; j < h.n_cols;++j)
				{
					for (int s = 0; s < h.n_slices;++s,++index)
					{
						hvec(index) = h(i, j, s);
					}
				}
			}

			arma::normalise(hvec, 2);
			for (int i = 0; i < hvec.n_elem;++i)
			{
				if (hvec(i)>0.2)
				{
					hvec(i) = 0.2;
				}
			}
			arma::normalise(hvec, 2);
			for (int i = 0; i < hvec.n_elem;++i)
			{
				thetahist.push_back(hvec(i));
			}

			//phi
			arma::fmat wz2 = weigth_ori_phi.submat(rindex, cindex, arma::size(blockSize.width, blockSize.height));

			compute_trilinear(wz2, spatialweights, phiweights);

			f = thetaweigths;
			m = gaussianweights;

			//interpolate magnitudes for binning
			mx1y1z1 = m%f.x1_y1_z1;
			mx1y1z2 = m%f.x1_y1_z2;
			mx2y1z1 = m%f.x2_y1_z1;
			mx2y1z2 = m%f.x2_y1_z2;
			mx1y2z1 = m%f.x1_y2_z1;
			mx1y2z2 = m%f.x1_y2_z2;
			mx2y2z1 = m%f.x2_y2_z1;
			mx2y2z2 = m%f.x2_y2_z2;

			orientationBins = binindexphi.submat(rindex, cindex, arma::size(blockSize.width, blockSize.height));

			//initialize block histogram to zero
			blockhistphi.zeros();

			for (int x = 0; x < blockSize.width; ++x)
			{
				int cx = spatialweights.cellX(x);
				for (int y = 0; y < blockSize.height; ++y)
				{
					int cy = spatialweights.cellY(y);
					int z = orientationBins(y, x);

					blockhistphi(cy, cx, z) = blockhistphi(cy, cx, z) + mx1y1z1(y, x);
					blockhistphi(cy, cx, z + 1) = blockhistphi(cy, cx, z + 1) + mx1y1z2(y, x);
					blockhistphi(cy + 1, cx, z) = blockhistphi(cy + 1, cx, z) + mx1y2z1(y, x);
					blockhisttheta(cy + 1, cx, z + 1) = blockhistphi(cy + 1, cx, z + 1) + mx1y2z2(y, x);
					blockhistphi(cy, cx + 1, z) = blockhistphi(cy, cx + 1, z) + mx2y1z1(y, x);
					blockhistphi(cy, cx + 1, z + 1) = blockhistphi(cy, cx + 1, z + 1) + mx2y1z2(y, x);
					blockhistphi(cy + 1, cx + 1, z) = blockhistphi(cy + 1, cx + 1, z) + mx2y2z1(y, x);
					blockhistphi(cy + 1, cx + 1, z + 1) = blockhistphi(cy + 1, cx + 1, z + 1) + mx2y2z2(y, x);
				}
			}

			//wrap orientation bins
			blockhistphi.slice(1) = blockhistphi.slice(1) + blockhistphi.slice(blockhistphi.n_slices - 1);
			blockhistphi.slice(blockhistphi.n_slices - 2) = blockhistphi.slice(blockhistphi.n_slices - 2) + blockhistphi.slice(0);

			arma::fcube p = blockhistphi(arma::span(1, blockhistphi.n_rows - 2), arma::span(1, blockhistphi.n_cols - 2),
				arma::span(1, blockhistphi.n_slices - 2));

			//converttovec
			pvec.zeros();
			for (int i = 0, index = 0; i < p.n_rows; ++i)
			{
				for (int j = 0; j < p.n_cols; ++j)
				{
					for (int s = 0; s < p.n_slices; ++s, ++index)
					{
						pvec(index) = p(i, j, s);
					}
				}
			}

			arma::normalise(pvec, 2);
			for (int i = 0; i < pvec.n_elem; ++i)
			{
				if (pvec(i) > 0.2)
				{
					pvec(i) = 0.2;
				}
			}
			arma::normalise(pvec, 2);

			for (int i = 0; i < pvec.n_elem; ++i)
			{
				phihist.push_back(pvec(i));
			}
		}
	}
}

void HONV::compute_hist(arma::fmat& theta, arma::fmat& phi, vector<float>& hist)
{
	//先不插值
	hist.reserve(numBlocksPerWin*bin_numphi*bin_numtheta*numCellsPerBlock);
	


}

void HONV::compute_wight()
{

	float sigma = 0.5*blockSize.width;
	float sigma2 = sigma*sigma;
	float sizew = (blockSize.width - 1) / 2;
	float sizeh = (blockSize.height - 1) / 2;

	gaussianweights.resize(blockSize.width, blockSize.height);
	//weights.create(blockSize);

#pragma omp parallel for
	for (int i = 0; i <= blockSize.height;++i)
	{
		for (int j = 0; j < blockSize.width;++j)
		{
			float dx = j - sizew;
			float dy = i - sizeh;
			gaussianweights(i,j) = exp(-(dx*dx + dy*dy) / (2 * sigma2));
		}
	}

	float sumweights = arma::accu(gaussianweights);
	
	gaussianweights = gaussianweights / sumweights;

	arma::fvec x(blockSize.width),y(blockSize.height);
	for (int i = 0; i < x.n_elem; ++i)
	{
		x(i) = 0.5 + i;
	}
	for (int i = 0; i < y.n_elem; ++i)
	{
		y(i) = 0.5 + i;
	}

	arma::fvec x1, y1;
	arma::ivec cellX1, cellY1;
	compute_lowerHistBin(x, cellSize.width, x1, cellX1);
	compute_lowerHistBin(y, cellSize.height, y1, cellY1);

	arma::fvec wx1 = 1 - (x - x1) / cellSize.width;
	arma::fvec wy1 = 1 - (y - y1) / cellSize.height;

	spatialweights.x1y1 = wy1*wx1.t();
	spatialweights.x2y1 = wy1*(1 - wx1).t();
	spatialweights.x1y2 = (1 - wy1)*wx1.t();
	spatialweights.x2y2 = (1 - wy1)*(1 - wx1).t();

	spatialweights.cellX = cellX1;
	spatialweights.cellY = cellY1;
}

void HONV::compute_lowerHistBin(arma::fmat& x, float binwidth,arma::fmat& bincenter,arma::imat& binindex)
{
	float invWidth = 1.0 / binwidth;
	arma::fmat bin = arma::floor(x / invWidth - 0.5);

	
	binindex= arma::conv_to<arma::imat >::from(bin + 1);
	bincenter = invWidth*(binindex + 0.5);
}

void HONV::compute_lowerHistBin(arma::fvec & x, float binwidth, arma::fvec & bincenter, arma::ivec & binindex)
{
	float invWidth = 1.0 / binwidth;
	arma::fvec bin = arma::floor(x / invWidth - 0.5);

	bincenter = invWidth*(bin + 1 + 0.5);
	binindex = arma::conv_to<arma::ivec >::from(bin + 1);

}




void HONV::compute_lowerHistBin(arma::fmat& theta, arma::fmat& phi, arma::fcube& bincenter, arma::icube& binindex)
{
	float binwidththeta = 2 * PI / bin_numtheta;
	float binwidthphi = 2 * PI / bin_numphi;

	//computeLowerHistBin
	arma::fmat bincentertheta, bincenterphi;
	arma::imat binindextheta, binindexphi;

	compute_HistBin(theta, binwidththeta, bincentertheta, binindextheta);
	compute_HistBin(phi, binwidthphi, bincenterphi, binindexphi);

	binindex.resize(theta.n_rows, theta.n_cols, 2);
	binindex.slice(0) = binindextheta;
	binindex.slice(1) = binindexphi;

	bincenter.resize(theta.n_rows, theta.n_cols, 2);
	bincenter.slice(0) = bincentertheta;
	bincenter.slice(1) = bincenterphi;
}

//without interplation
void HONV::compute_HistBin(arma::fmat& x, float binwidth, arma::fmat& bincenter, arma::imat& binindex)
{
	float invWidth = 1.0 / binwidth;
	arma::fmat bin = arma::floor(x / invWidth);

	bincenter = invWidth*(bin + 0.5);
	binindex = arma::conv_to<arma::imat >::from(bin + 1);
}

void HONV::compute_trilinear(arma::fmat& wz, spatialWeights& weights, trilinearWeights triweights)
{
	triweights.x1_y1_z1 = wz%weights.x1y1;
	triweights.x1_y1_z2 = weights.x1y1 - triweights.x1_y1_z1;
	triweights.x2_y1_z1 = wz%weights.x2y1;
	triweights.x2_y1_z2 = weights.x2y1 - triweights.x2_y1_z1;
	triweights.x1_y2_z1 = wz%weights.x1y2;
	triweights.x1_y2_z2 = weights.x1y2 - triweights.x1_y2_z1;
	triweights.x2_y2_z1 = wz%weights.x2y2;
	triweights.x2_y2_z2 = weights.x2y2 - triweights.x2_y2_z1;
}
