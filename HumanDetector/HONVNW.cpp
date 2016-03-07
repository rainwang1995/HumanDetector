#include "HONVNW.h"
#include <opencv2/core/operations.hpp>
#include <iterator>
#include <omp.h>
using namespace cv;

template<typename _Tp> static inline _Tp gcd(_Tp a, _Tp b)
{
	if (a < b)
		std::swap(a, b);
	while (b > 0)
	{
		_Tp r = a % b;
		a = b;
		b = r;
	}
	return a;
}

void HONVNW::compute(cv::Mat & img, vector<float>& feature)const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);
	//Mat dx8u;
	//dx8u = abs(dx);
	//dx8u.convertTo(dx8u, CV_8U);
	//imshow("dx", dx8u);
	////waitKey();

	//Mat dy8u;
	//dy8u = abs(dy);
	//dy8u.convertTo(dy8u, CV_8U);
	//imshow("dy", dy8u);
	//waitKey();
	arma::fmat theta, phi;
	compute_theta_phi(dx, dy, theta, phi);
	arma::fcube bincenter;
	arma::icube binindex;
	//cout << theta << endl;
	compute_HistBin(theta, phi, bincenter, binindex);

	arma::fvec hist;
	compute_hist(binindex, hist);

	feature = arma::conv_to<vector<float> >::from(hist);
}

void HONVNW::setSvmDetector(cv::Ptr<cv::ml::SVM>& _svm)
{
	//svm = cv::ml::SVM::create();
	svm = _svm;
}

void HONVNW::setSvmDetector(const string& xmlfile)
{
	svm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void HONVNW::detect(cv::Mat& img, vector<Point>& foundLocations, vector<double>& weights, double hisThreshold /*= 0*/, cv::Size winStride, const vector<Point>& locations)const
{
	foundLocations.clear();
	if (svm->empty())
	{
		return;
	}

	if (winStride==Size())
	{
		winStride = cellSize;
	}

	//Í¼ÏñÌî³ä
	Size stride(gcd(winStride.width, blockStride.width), gcd(winStride.height, blockStride.height));

	size_t nwindows = locations.size();
	Size padding;
	padding.width = (int)alignSize(std::max(padding.width, 0), stride.width);
	padding.height = (int)alignSize(std::max(padding.height, 0), stride.height);

	Size paddedImgSize(img.cols + padding.width * 2, img.rows + padding.height * 2);

	if (!nwindows)
	{
		nwindows = Size((paddedImgSize.width - winSize.width) / winStride.width + 1,
			(paddedImgSize.height - winSize.height) / winStride.height + 1).area();
	}
	
	Mat paddedimg;
	copyMakeBorder(img, paddedimg, padding.height, padding.height, padding.width, padding.width, BORDER_REFLECT_101);

	int numwinR = (paddedImgSize.width - winSize.width) / winStride.width + 1;
	int numwinC = (paddedImgSize.height - winSize.height) / winStride.height + 1;

	if (locations.size())
	{
		Point pt0;
		for (int i = 0; i < nwindows;++i)
		{
			pt0 = locations[i];
			pt0 = locations[i];
			if (pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
				pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height)
				continue;

			Point pt(pt0.x + padding.width, pt0.y + padding.height);
			
			Mat winimg = paddedimg(Rect(pt.x, pt.y, winSize.width, winSize.height));

			vector<float> feature;
			compute(winimg, feature);

			float response = svm->predict(feature);
			//cout << response << endl;
			if ((int)response == 1)
			{
				foundLocations.push_back(pt0);
				weights.push_back(response);
			}
		}
	}

	else
	{
		//scan over windows
		for (int j = 0; j < numwinC; ++j)
		{
			for (int i = 0; i < numwinR; ++i)
			{
				Point pt0;

				Mat winimg = paddedimg(Rect(i*winStride.width,j*winStride.height, winSize.width, winSize.height));

				pt0.x = i*winStride.width - padding.width;
				pt0.y = j*winStride.height - padding.height;

				vector<float> feature;
				compute(winimg, feature);
				Mat result;
				//float response = svm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
				float response = svm->predict(feature);
				//cout << response << endl;
				if ((int)response==1)
				{
					foundLocations.push_back(pt0);
					weights.push_back(response);
				}
				/*if (response >= hisThreshold)
				{
					foundLocations.push_back(pt0);
					weights.push_back(response);
				}*/
			}
		}
	}
	
	
}

class Parallel_Detection :public ParallelLoopBody
{
private:
	const HONVNW* honv;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection(const HONVNW* _honv, Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights=0, vector<double>* _scales=0)
	{
		honv = _honv;
		img = _img;
		hitThreshold = _hitThreshold;
		winStride = _winStride;
		levelScale = _levelScale;
		mtx = _mtx;
		vec = _vec;
		weights = _weights;
		scales = _scales;
	}

	void operator() (const Range& range) const
	{
		int i, i1 = range.start, i2 = range.end;
		double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1 + 1] : std::max(img.cols, img.rows);
		Size maxSz(cvCeil(img.cols / minScale), cvCeil(img.rows / minScale));
		Mat smallerImgBuf(maxSz, img.type());
		vector<Point> locations;
		vector<double> hitsWeights;

		for (i = i1; i < i2; i++)
		{
			double scale = levelScale[i];
			Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
			Mat smallerImg(sz, img.type(), smallerImgBuf.data);
			if (sz == img.size())
				smallerImg = Mat(sz, img.type(), img.data, img.step);
			else
				resize(img, smallerImg, sz,0.0,0.0,INTER_NEAREST);
			honv->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(honv->winSize.width*scale), cvRound(honv->winSize.height*scale));
			
			mtx->lock();
			
			for (size_t j = 0; j < locations.size(); j++)
			{
				/*cout << "scale: " << scale << endl;
				cout << "locations: " << cvRound(locations[j].x*scale) << " " << cvRound(locations[j].y*scale) << endl;
				cout << "scaled WinSize: " << scaledWinSize.width << " " << scaledWinSize.height << endl;*/
				vec->push_back(Rect(cvRound(locations[j].x*scale),
					cvRound(locations[j].y*scale),
					scaledWinSize.width, scaledWinSize.height));
				if (scales)
				{
					scales->push_back(scale);
				}
			}
			mtx->unlock();

			if (weights && (!hitsWeights.empty()))
			{
				mtx->lock();
				for (size_t j = 0; j < locations.size(); j++)
				{
					weights->push_back(hitsWeights[j]);
				}
				mtx->unlock();
			}
		}
	}
};



void HONVNW::detectMultiScale(cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double scale0 /*= 1.05*/, double finalThreshold,bool usemeanshift)const
{
	if (svm->empty())
	{
		cerr << "svm error" << endl;
		return;
	}
	double scale = 1.;
	int levels = 0;
	vector<double> levelScale;
	for (levels = 0; levels < nlevels;++levels)
	{
		levelScale.push_back(scale);
		if (cvRound(img.cols/scale)<winSize.width||cvRound(img.rows/scale)<winSize.height
			||scale0<=1)
		{
			break;
		}

		scale *= scale0;
	}

	levels = std::max(levels, 1);
	levelScale.resize(levels);

	vector<Rect> allCandidates;
	vector<double> tempScales;
	vector<double> tempWeights;
	vector<double> foundScales;

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection(this, img, hitThreshold, winStride, &levelScale[0], &allCandidates, &mtx, &tempWeights, &tempScales));

	foundScales.clear();
	std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
	foundlocations.clear();
	std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundlocations));
	weights.clear();
	std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(weights));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}

	//debug
	//cout << "before detect function:" << endl;
	//for (int j = 0; j < foundlocations.size(); ++j)
	//{
	//	Rect r = foundlocations[j];
	//	cout << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
	//	if (r.x < 0)
	//		r.x = 0;
	//	if (r.y < 0)
	//		r.y = 0;
	//	if (r.x + r.width > img.cols)
	//		r.width = img.cols - r.x;
	//	if (r.y + r.height > img.rows)
	//		r.height = img.rows - r.y;
	//}
}

void HONVNW::cal_para()
{
	cellnumR = winSize.width / cellSize.width;
	cellnumC = winSize.height / cellSize.height;

	cellnumRblock = blockSize.width / cellSize.width;
	cellnumCblock = blockSize.height / cellSize.height;

	blockStride.width = blockSize.width;
	blockStride.height = blockSize.height;

	numCellsPerBlock = blockSize.width / cellSize.width*blockSize.height / cellSize.height;

	blocknumR = winSize.width / blockStride.width ;
	blocknumC = winSize.height / blockStride.height;
	numBlocksPerWin = blocknumR*blocknumC;

	featureLen = numCellsPerBlock*bin_numphi*bin_numtheta*numBlocksPerWin;

	maskdx=Mat::zeros(1, HONV_difference * 2 + 1, CV_32F);

	maskdy=Mat::zeros(HONV_difference * 2 + 1, HONV_difference * 2 + 1, CV_32F);
	maskdx.at<float>(0, 0) = -0.5;
	maskdx.at<float>(0, maskdx.cols - 1) = 0.5;

	maskdy.at<float>(0, HONV_difference) = -0.5;
	maskdy.at<float>(maskdy.rows-1, HONV_difference) = 0.5;
	//cout << maskdy << endl;
	//cout << maskdx << endl;
}

void HONVNW::compute_dxdy(cv::Mat & src, cv::Mat & dximg, cv::Mat & dyimg)const
{
	filter2D(src, dximg, maskdx.depth(), maskdx);
	filter2D(src, dyimg, maskdy.depth(), maskdy);
	//cout << dyimg << endl;
	//cout << dximg;
}

void HONVNW::compute_theta_phi(cv::Mat & dximg, cv::Mat & dyimg, arma::Mat<float>& theta, arma::Mat<float>& phi)const
{
	arma::Mat<float> dxmat(dximg.ptr<float>(0), dximg.rows, dximg.cols, false);
	arma::Mat<float> dymat(dyimg.ptr<float>(0), dyimg.rows, dyimg.cols, false);

	arma::Mat<float> powmatx = arma::pow(dxmat, 2.0);
	arma::Mat<float> powmaty = arma::pow(dymat, 2.0);

	phi = arma::atan(dymat / dxmat) + PI / 2;
	//cout << phi << endl;
#pragma omp parallel for
	for (int j = 0; j < phi.n_rows;++j)
	{
		for (int i = 0; i < phi.n_cols;++i)
		{
			if (arma::arma_isnan(phi(j,i)))
			{
				phi(j, i) = 0;
			}
		}
	}
	//cout << phi;
	theta = arma::atan(arma::sqrt(powmatx + powmaty)) + PI / 2;
	//cout << theta;
}

void HONVNW::compute_HistBin(arma::fmat& theta, arma::fmat& phi, arma::fcube& bincenter, arma::icube& binindex)const
{
	float thetabinwidth = PI / bin_numtheta;
	float phibinwidth = PI / bin_numphi;

	//computeLowerHistBin
	arma::fmat bincentertheta, bincenterphi;
	arma::imat binindextheta, binindexphi;

	compute_HistBin(theta, thetabinwidth, bincentertheta, binindextheta);
	compute_HistBin(phi, phibinwidth, bincenterphi, binindexphi);

#pragma omp parallel for
	for (int i = 0; i < binindextheta.n_rows;++i)
	{
		for (int j = 0; j < binindextheta.n_cols;++j)
		{
			if (binindextheta(i,j)>=bin_numtheta)
			{
				binindextheta(i, j) = bin_numtheta - 1;
			}
		}
	}

#pragma omp parallel for
	for (int i = 0; i < binindexphi.n_rows; ++i)
	{
		for (int j = 0; j < binindexphi.n_cols; ++j)
		{
			if (binindexphi(i, j) >= bin_numphi)
			{
				binindexphi(i, j) = bin_numphi - 1;
			}
		}
	}

	binindex.resize(theta.n_rows, theta.n_cols, 2);
	//binindex.zeros();
	//cout << binindexphi;

	binindex.slice(0) = binindextheta;
	binindex.slice(1) = binindexphi;

	bincenter.resize(theta.n_rows, theta.n_cols, 2);
	bincenter.slice(0) = bincentertheta;
	bincenter.slice(1) = bincenterphi;

}

void HONVNW::compute_HistBin(arma::fmat& x, float binwidth, arma::fmat& bincenter, arma::imat& binindex)const
{
	//float invWidth = 1.0 / binwidth;
	arma::fmat bin = arma::floor(x / binwidth);

	binindex = arma::conv_to<arma::imat >::from(bin);
	bincenter = binwidth*(bin + 0.5);
}

void HONVNW::compute_hist_cell(arma::icube& binindex, arma::ivec& cellhist)const
{
	arma::imat cellmat(bin_numtheta, bin_numphi, arma::fill::zeros);

	for (int y = 0; y < binindex.n_rows;++y)
	{
		for (int x = 0; x < binindex.n_cols;++x)
		{
			/*cout << binindex(y, x, 0) << " " << binindex(y, x, 1) << endl;
			cout << bin_numphi << " " << bin_numtheta << endl;
			cout << cellmat.n_rows << " " << cellmat.n_cols << endl;*/
			cellmat(binindex(y, x, 0), binindex(y, x, 1)) = cellmat(binindex(y, x, 0), binindex(y, x, 1)) + 1;
		}
	}

	cellhist=arma::vectorise(cellmat);
	//cellhist = arma::conv_to<arma::ivec>::from(cellmat);
}

void HONVNW::compute_hist_block(arma::icube & binindex, arma::fvec& blockhist)const
{
	blockhist.resize(numCellsPerBlock*bin_numtheta*bin_numphi);

	for (int y = 0,p=0; y < cellnumCblock;++y)
	{
		for (int x = 0; x < cellnumRblock;++x,++p)
		{
			arma::icube cell = binindex.subcube(y*cellSize.height, x*cellSize.width, 0, 
				arma::SizeCube(cellSize.height, cellSize.width, 2));

			arma::ivec cellhist;
			compute_hist_cell(cell, cellhist);
			int startindex = p*cellhist.n_elem;
			for (int i = 0; i < cellhist.n_elem;++i)
			{
				blockhist(startindex + i) = cellhist(i);
			}
		}
	}

	//¹éÒ»»¯
	arma::normalise(blockhist, 2);
	for (int i = 0; i < blockhist.n_elem;++i)
	{
		if (blockhist(i)>NORMTHR)
		{
			blockhist(i) = NORMTHR;
		}
	}

	arma::normalise(blockhist, 2);
}

void HONVNW::compute_hist(arma::icube & binindex, arma::fvec & hist)const
{
	hist.resize(featureLen);
	
	//scan all blocks
	for (int y = 0,p=0; y < blocknumC;++y)
	{
		for (int x = 0; x < blocknumR;++x,++p)
		{
			arma::icube blockcube = binindex.subcube(y*blockStride.height, 
				x*blockStride.width, 0, arma::SizeCube(blockStride.height, blockStride.width, 2));

			arma::fvec blockhist;
			compute_hist_block(blockcube, blockhist);

			int startindex = p*blockhist.n_elem;
			for (int i = 0; i < blockhist.n_elem;++i)
			{
				hist(startindex + i) = blockhist(i);
			}
		}
	}

}

void HONVNW::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
{
	if (groupThreshold <= 0 || rectList.empty())
	{
		return;
	}

	CV_Assert(rectList.size() == weights.size());

	vector<int> labels;
	int nclasses = partition(rectList, labels, SimilarRects(eps));

	vector<cv::Rect_<double> > rrects(nclasses);
	vector<int> numInClass(nclasses, 0);
	vector<double> foundWeights(nclasses, -std::numeric_limits<double>::max());
	int i, j, nlabels = (int)labels.size();

	for (i = 0; i < nlabels; i++)
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		foundWeights[cls] = max(foundWeights[cls], weights[i]);
		numInClass[cls]++;
	}

	for (i = 0; i < nclasses; i++)
	{
		// find the average of all ROI in the cluster
		cv::Rect_<double> r = rrects[i];
		double s = 1.0 / numInClass[i];
		rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
			cv::saturate_cast<double>(r.y*s),
			cv::saturate_cast<double>(r.width*s),
			cv::saturate_cast<double>(r.height*s));
	}

	rectList.clear();
	weights.clear();

	for (i = 0; i < nclasses; i++)
	{
		cv::Rect r1 = rrects[i];
		int n1 = numInClass[i];
		double w1 = foundWeights[i];
		if (n1 <= groupThreshold)
			continue;
		// filter out small rectangles inside large rectangles
		for (j = 0; j < nclasses; j++)
		{
			int n2 = numInClass[j];

			if (j == i || n2 <= groupThreshold)
				continue;

			cv::Rect r2 = rrects[j];

			int dx = cv::saturate_cast<int>(r2.width * eps);
			int dy = cv::saturate_cast<int>(r2.height * eps);

			if (r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3))
				break;
		}

		if (j == nclasses)
		{
			rectList.push_back(r1);
			weights.push_back(w1);
		}
	}
}

