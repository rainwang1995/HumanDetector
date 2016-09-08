#include "HONVNW.h"
#include <iterator>
#include <omp.h>
#include <cmath>
using namespace cv;

void HONVNW::compute(const cv::Mat & img, vector<float>& feature)const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);
	Mat theta, phi;

	compute_theta_phi(dx, dy, theta, phi);
	Mat bincenter;
	Mat binindex;
	//cout << theta << endl;
	compute_HistBin(theta, phi, bincenter, binindex);
	compute_hist(binindex, feature);
}

void HONVNW::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	//svm = cv::ml::SVM::create();
	svm = _svm;
}

void HONVNW::loadSvmDetector(const string & xmlfile)
{
	svm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void HONVNW::detect(const cv::Mat& img, vector<Point>& foundLocations, vector<double>& weights, double hisThreshold /*= 0*/, cv::Size winStride, const vector<Point>& locations)const
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

	//¼ÆËãÕû·ùÍ¼
	Mat binindeximg;
	compute_win(paddedimg, binindeximg);

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

				Rect rt = Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height);

				//Mat winimg = paddedimg(rt);

				Mat binindexroi = binindeximg(rt);

				pt0.x = i*winStride.width - padding.width;
				pt0.y = j*winStride.height - padding.height;

				vector<float> feature;
				//compute(winimg, feature);
				compute_hist(binindexroi, feature);

				Mat result;
				float response = svm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
				response = result.at<float>(0, 0);
				if (response <= -hisThreshold)
				{
					foundLocations.push_back(pt0);
					weights.push_back(-response);
				}
			}
		}
	}
	
	
}

void HONVNW::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_HONV :public ParallelLoopBody
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
	Parallel_Detection_HONV(const HONVNW* _honv, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
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
		//cout << "honv" << endl;
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


void HONVNW::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (svm->empty())
	{
		cerr << "svm error" << endl;
		return;
	}
	double scale = 1.;
	scale0 = 1.1;
	int levels = 0;
	vector<double> levelScale;
	for (levels = 0; levels < nlevels; ++levels)
	{
		levelScale.push_back(scale);
		if (cvRound(img.cols / scale) < winSize.width || cvRound(img.rows / scale) < winSize.height
			|| scale0 <= 1)
		{
			break;
		}

		scale *= scale0;
	}

	levels = std::max(levels, 1);
	levelScale.resize(levels);

	foundlocations.clear();
	weights.clear();
	/*vector<Rect> allCandidates;
	vector<double> tempScales;
	vector<double> tempWeights;*/
	vector<double> foundScales;

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection_HONV(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void HONVNW::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/,
	double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
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

void HONVNW::compute_dxdy(const cv::Mat & src, cv::Mat & dximg, cv::Mat & dyimg)const
{
	filter2D(src, dximg, maskdx.depth(), maskdx);
	filter2D(src, dyimg, maskdy.depth(), maskdy);
	//cout << dyimg << endl;
	//cout << dximg;
}

void HONVNW::compute_theta_phi(cv::Mat & dximg, cv::Mat & dyimg, Mat& theta, Mat& phi)const
{

	Mat powmatx = dximg.mul(dximg);
	Mat powmaty = dyimg.mul(dyimg);

	theta.create(dximg.size(), CV_32FC1);
	phi.create(dyimg.size(), CV_32FC1);

	for (int j = 0; j < phi.rows;++j)
	{
		const float* dximgPtr = dximg.ptr<float>(j);
		const float* dyimgPtr = dyimg.ptr<float>(j);

		float* phiPtr = phi.ptr<float>(j);
		float* thetaPtr = theta.ptr<float>(j);
		for (int i = 0; i < phi.cols;++i)
		{
			float dx = dximgPtr[i];
			float dy = dyimgPtr[i];

			phiPtr[i] = 0;
			float phiangle = 0;
			if (abs(dx)>1e-6)
			{
				phiPtr[i] = atan(dy / dx) + PI / 2;
			}

			float sqrtxy = sqrt(dx*dx + dy*dy);
			thetaPtr[i] = atan(sqrtxy) + PI / 2;
		}
	}
}

void HONVNW::compute_HistBin(Mat& theta, Mat& phi,Mat& bincenter, Mat& binindex)const
{
	float thetabinwidth = PI / bin_numtheta;
	float phibinwidth = PI / bin_numphi;

	//computeLowerHistBin
	Mat bincentertheta, bincenterphi;
	Mat binindextheta, binindexphi;

	compute_HistBin(theta, thetabinwidth, bincentertheta, binindextheta);
	compute_HistBin(phi, phibinwidth, bincenterphi, binindexphi);

//#pragma omp parallel for
	for (int i = 0; i < binindextheta.rows;++i)
	{
		uchar* ptr = binindextheta.ptr<uchar>(i);
		for (int j = 0; j < binindextheta.cols;++j)
		{
			if (ptr[j]>=bin_numtheta)
			{
				ptr[j]= 0;
			}
		}
	}

//#pragma omp parallel for
	for (int i = 0; i < binindexphi.rows; ++i)
	{
		uchar* ptr = binindextheta.ptr<uchar>(i);
		for (int j = 0; j < binindexphi.cols; ++j)
		{
			if (ptr[j] >= bin_numphi)
			{
				ptr[j] = 0;
			}
		}
	}

	vector<Mat> channels(2);
	channels[0] = binindextheta;
	channels[1] = binindexphi;

	merge(channels, binindex);

	vector<Mat> channelscenter(2);
	channelscenter[0] = bincentertheta;
	channelscenter[1] = bincenterphi;
	merge(channelscenter, bincenter);

}

void HONVNW::compute_HistBin(Mat& x, float binwidth, Mat& bincenter, Mat& binindex)const
{
	//float invWidth = 1.0 / binwidth;
	Mat bin(x.size(), CV_8UC1);
	for (int i = 0; i < x.rows;++i)
	{
		float* xPtr = x.ptr<float>(i);
		uchar* bPtr = bin.ptr<uchar>(i);
		for (int j = 0; j < x.cols;++j)
		{
			bPtr[j] = floor(xPtr[j] / binwidth);
		}
	}
	binindex = bin;
	Mat binf = bin + 0.5;
	bincenter = binwidth*binf;
}

void HONVNW::compute_hist_cell(Mat& binindex, float* cellhist)const
{
	for (int y = 0; y < binindex.rows;++y)
	{
		uchar* ptr = binindex.ptr<uchar>(y);
		for (int x = 0; x < binindex.cols;++x)
		{			
			int index = ptr[2 * x] * bin_numphi + ptr[2 * x + 1];
			++cellhist[index];		
		}
	}
}

void HONVNW::compute_hist_block(Mat & binindex, float* blockhist)const
{
	for (int y = 0,p=0; y < cellnumCblock;++y)
	{
		uchar* indexptr = binindex.ptr<uchar>(y);
		for (int x = 0; x < cellnumRblock;++x,++p)
		{	
			Mat subcell = binindex(Rect(x*cellSize.width, y*cellSize.height, cellSize.width, cellSize.height));
			compute_hist_cell(subcell, &blockhist[bin_numphi*bin_numtheta*p]);
		}
	}

	
	//¹éÒ»»¯
	normliseblock(blockhist);
}

void HONVNW::normliseblock(float* blockhist) const
{
	float* hist = &blockhist[0];
	size_t i, sz = bin_numtheta*bin_numtheta*numCellsPerBlock;

	float sum = 0;
	for (i = 0; i < sz; i++)
		sum += hist[i] * hist[i];

	float scale = 1.f / (std::sqrt(sum) + sz*0.1f), thresh = 0.2;

	for (i = 0, sum = 0; i < sz; i++)
	{
		hist[i] = std::min(hist[i] * scale, thresh);
		sum += hist[i] * hist[i];
	}

	scale = 1.f / (std::sqrt(sum) + 1e-3f);

	for (i = 0; i < sz; i++)
		hist[i] *= scale;
}

void HONVNW::compute_hist(Mat& binindex, vector<float>& hist)const
{
	hist.clear();
	hist.resize(featureLen);
	int featurelenPerblock = numCellsPerBlock*bin_numphi*bin_numtheta;
	//scan all blocks
	for (int y = 0,p=0; y < blocknumC;++y)
	{
		for (int x = 0; x < blocknumR;++x,++p)
		{
			Mat blockmat = binindex(Rect(x*blockStride.width, y*blockStride.height, blockSize.width, blockSize.height));

			compute_hist_block(blockmat, &hist[p*featurelenPerblock]);
		}
	}

}

void HONVNW::compute_win(Mat& src, Mat& binindex) const
{
	Mat dx, dy;
	compute_dxdy(src, dx, dy);
	Mat theta, phi;

	compute_theta_phi(dx, dy, theta, phi);
	Mat bincenter;
	//cout << theta << endl;
	compute_HistBin(theta, phi, bincenter, binindex);
}

void HONVNW::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
{
	if (groupThreshold <= 0 || rectList.empty()||rectList.size()==1)
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

