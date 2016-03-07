#include "SLTP.h"
#include <iterator>
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

void SLTP::compute(const Mat& img, vector<float>& features) const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signdx, signdy;
	compute_sign(dx, signdx);
	compute_sign(dy, signdy);

	compute_histwin(signdx, signdy, features);
}

void SLTP::setDefaultParams()
{
	winSize = Size(64, 128);
	cellSize = Size(8, 8);
	signThreshold = 30;	
	//nlevels = 64;
	//scale0 = 1.05;
}

void SLTP::cal_params()
{
	numCellC = winSize.height / cellSize.height;
	numCellR = winSize.width / cellSize.width;

	numCellPerWin = numCellR*numCellC;

	maskdx = Mat::zeros(3, 3, CV_8SC1);
	maskdy - Mat::zeros(3, 3, CV_8SC1);

	//set mask
	maskdx.at<char>(1, 0) = -1;
	maskdx.at<char>(1, 2) = 1;
	maskdy.at<char>(0, 1) = -1;
	maskdy.at<char>(2, 1) = 1;

	//compute feature len
	featurelen = 9 * numCellPerWin;

	//set sign array
	for (int i = 0,p=0; i < 3;++i)
	{
		for (int j = 0; j < 3; ++j, ++p)
		{
			signarry[i][j] = p;
		}
	}
}

void SLTP::compute_dxdy(const Mat& img, Mat& dx, Mat& dy) const
{
	filter2D(img, dx, img.depth(), maskdx);
	filter2D(img, dy, img.depth(), maskdy);
}

void SLTP::compute_sign(const Mat& dimg, Mat& signimg) const
{
	signimg=Mat::zeros(dimg.size(), CV_8SC1);

	signimg.setTo(-1, dimg <= -signThreshold);
	signimg.setTo(1, dimg >= signThreshold);
}

void SLTP::compute_histcell(const Mat& signimgx, const Mat& signimgy, vector<float>& hist) const
{
	CV_Assert(signimgx.rows == signimgy.rows&&signimgx.cols==signimgy.cols);
	hist.clear();
	hist.resize(9,0);

	for (int i = 0;i<signimgx.rows;++i)
	{
		for (int j = 0; j < signimgx.cols;++j)
		{
			int x = signimgx.at<char>(i, j);
			int y = signimgy.at<char>(i, j);
			int mode = signarry[x + 1][y + 1];

			++hist[mode];
		}
	}
}

void SLTP::compute_histcell(const Mat& signimgx, const Mat& signimgy, float* hist) const
{
	CV_Assert(signimgx.rows == signimgy.rows&&signimgx.cols == signimgy.cols);

	for (int i = 0; i < signimgx.rows; ++i)
	{
		for (int j = 0; j < signimgx.cols; ++j)
		{
			int x = signimgx.at<char>(i, j);
			int y = signimgy.at<char>(i, j);
			int mode = signarry[x + 1][y + 1];

			++hist[mode];
		}
	}
}

void SLTP::compute_histwin(const Mat& signimgx, const Mat& signimgy, vector<float>& hist) const
{
	CV_Assert(signimgx.rows == signimgy.rows&&signimgx.cols == signimgy.cols);
	hist.clear();
	hist.resize(9*numCellPerWin, 0);

	//scan all cells
	for (int i = 0; i < numCellC;++i)
	{
		for (int j = 0; j < numCellR; ++j)
		{
			Rect cellrt(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height);

			Mat cellx = signimgx(cellrt);
			Mat celly = signimgy(cellrt);

			float* histptr = &hist[9 * (numCellR*i + j)];
			compute_histcell(cellx, celly, histptr);
		}
	}
}

void SLTP::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
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

void SLTP::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	sltpsvm = _svm;
}

void SLTP::setSvmDetector(const string& xmlfile)
{
	sltpsvm = ml::StatModel::load<ml::SVM>(xmlfile);
}


void SLTP::detect(const Mat& img, vector<Point>& foundlocations, cv::Size winStride/*=cv::Size()*/, vector<cv::Point>& locations/*=vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (sltpsvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}

	if (winStride == Size())
	{
		winStride = cellSize;
	}

	//ͼ�����
	Size stride(gcd(winStride.width, cellSize.width), gcd(winStride.height, cellSize.height));

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
	if (img.size() != paddedimg.size())
	{
		copyMakeBorder(img, paddedimg, padding.height, padding.height, padding.width, padding.width, BORDER_REFLECT_101);
	}
	else
		paddedimg = img;

	int numwinR = (paddedImgSize.width - winSize.width) / winStride.width + 1;
	int numwinC = (paddedImgSize.height - winSize.height) / winStride.height + 1;

	if (locations.size())
	{
		Point pt0;
		for (int i = 0; i < nwindows; ++i)
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

			float response = sltpsvm->predict(feature);
			//cout << response << endl;
			if ((int)response == 1)
			{
				foundlocations.push_back(pt0);
				//weights.push_back(response);
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

				Mat winimg = paddedimg(Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height));

				pt0.x = i*winStride.width - padding.width;
				pt0.y = j*winStride.height - padding.height;

				vector<float> feature;
				compute(winimg, feature);
				Mat result;
				//float response = svm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
				float response = sltpsvm->predict(feature);
				//cout << response << endl;
				if ((int)response == 1)
				{
					foundlocations.push_back(pt0);
					//weights.push_back(response);
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
	const SLTP* sltp;
	Mat img;
	//double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection(const SLTP* _sltp, const Mat& _img, /*double _hitThreshold,*/ Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		sltp = _sltp;
		img = _img;
		//hitThreshold = _hitThreshold;
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
				resize(img, smallerImg, sz, 0.0, 0.0, INTER_NEAREST);
			sltp->detect(smallerImg, locations, /*hitsWeights, hitThreshold, */winStride);
			Size scaledWinSize = Size(cvRound(sltp->winSize.width*scale), cvRound(sltp->winSize.height*scale));

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

void SLTP::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, cv::Size winStride /*= cv::Size()*/, double nlevels/*=64*/, double scale0 /*= 1.05*/, double hitThreshold /*= 0*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (sltpsvm->empty())
	{
		cerr << "svm error" << endl;
		return;
	}
	double scale = 1.;
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

	vector<Rect> allCandidates;
	vector<double> tempScales;
	vector<double> tempWeights;
	vector<double> foundScales;

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection(this, img,/* hitThreshold,*/ winStride, &levelScale[0], &allCandidates, &mtx, &tempWeights, &tempScales));

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
}