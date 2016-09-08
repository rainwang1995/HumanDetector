#include "SLTP3.h"
#include <iterator>
using namespace cv;

void SLTP3::compute(const Mat& img, vector<float>& features) const
{
	CV_Assert(img.size() == winSize);

	Mat modemap;
	compute_modemap(img, modemap);

	compute_histimg(modemap, features);
}

void SLTP3::setDefaultParams()
{
	winSize = Size(64, 128);
	cellSize = Size(8, 8);
	signThreshold = 30;
	//nlevels = 64;
	//scale0 = 1.05;
}

void SLTP3::cal_params()
{
	numCellC = winSize.height / cellSize.height;
	numCellR = winSize.width / cellSize.width;

	numCellPerWin = numCellR*numCellC;

	//maskdx = Mat::zeros(3, 3, CV_8SC1);
	//maskdy = Mat::zeros(3, 3, CV_8SC1);

	//maskdx= (Mat_<char>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
	//maskdy = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);

	maskdx = (Mat_<char>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
	maskdy = (Mat_<char>(3, 3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);

	//compute feature len
	featurelen = 9 * numCellPerWin;

	//set sign array
	for (int i = 0, p = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j, ++p)
		{
			signarry[i][j] = p;
		}
	}
}

void SLTP3::compute_dxdy(const Mat& img, Mat& dx, Mat& dy) const
{
	filter2D(img, dx, CV_32FC1, maskdx);
	filter2D(img, dy, CV_32FC1, maskdy);
}

void SLTP3::compute_sign(const Mat& dimg, Mat& signimg, int thrpos,int thrneg)const
{
	signimg = Mat::zeros(dimg.size(), CV_8SC1);

	signimg.setTo(-1, dimg <= -thrneg);
	signimg.setTo(1, dimg >= thrpos);
}


void SLTP3::compute_histcell(const cv::Mat& modelmap, float* hist)const
{
	const char* ptrx;
	for (int i = 0; i < modelmap.rows; ++i)
	{
		ptrx = modelmap.ptr<char>(i);
		for (int j = 0; j < modelmap.cols; ++j)
		{
			int mode = ptrx[j];

			++hist[mode];
		}
	}
}

void SLTP3::compute_histimg(const cv::Mat& modemap, vector<float>& features) const
{
	features.clear();

	int numCellcols = modemap.cols / cellSize.width;
	int numCellrows = modemap.rows / cellSize.height;

	features.resize(9 * numCellcols*numCellrows, 0);
	//scan all cells
	//#pragma omp parallel
	{
		//Mat cellx, celly;
		for (int i = 0; i < numCellrows; ++i)
		{
			for (int j = 0; j < numCellcols; ++j)
			{
				Rect cellrt(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height);

				Mat modelcell = modemap(cellrt);

				float* histptr = &features[9 * (numCellcols*i + j)];
				compute_histcell(modelcell,  histptr);
			}
		}
	}
}

void SLTP3::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
{
	if (groupThreshold <= 0 || rectList.empty() || rectList.size() == 1)
		//if (groupThreshold <= 0 || rectList.empty())

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

void SLTP3::get_winfeature(vector<float>& featuresimg, vector<float>& featureswin, cv::Point& startpos, int cellimgcols) const
{
	featureswin.clear();
	featureswin.resize(featurelen, 0);

	int cellfeaturelen = 9 * numCellR;
	for (int i = 0; i < numCellC; ++i)
	{
		int sindex = cellimgcols*(startpos.y + i) + startpos.x;
		sindex *= 9;
		memcpy(&featureswin[i*cellfeaturelen], &featuresimg[sindex], sizeof(float)*cellfeaturelen);
	}
}



void SLTP3::compute_thr(const cv::Mat& imgd, cv::Mat& thrpos, cv::Mat& thrneg)const
{
	int numCellcols = imgd.cols / cellSize.width;
	int numCellrows = imgd.rows / cellSize.height;

	thrpos = Mat::zeros(numCellrows, numCellcols, CV_32S);
	thrneg = Mat::zeros(numCellrows, numCellcols, CV_32S);

	Mat celldx,posdx;

	posdx = Mat::zeros(cellSize, CV_32FC1);
	for (int i = 0; i < numCellrows;++i)
	{
		for (int j = 0; j < numCellcols;++j)
		{
			Rect cellrt(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height);

			celldx = imgd(cellrt);
			//posdx = Mat::zeros(celldx.size(), CV_32FC1);
			posdx.setTo(0);
			celldx.copyTo(posdx, celldx > 0);
			double sumposx = sum(posdx)[0];
			double sumx = sum(abs(celldx))[0];
			double sumnegx = sumx - sumposx;

			int thxpos = round(sumposx / (celldx.size().area()));
			int thxneg = round(sumnegx / (celldx.size().area()));
			
			thrpos.at<int>(i, j) = thxpos;
			thrneg.at<int>(i, j) = thxneg;
		}
	}
}

void SLTP3::compute_signimg(const cv::Mat& imgd, cv::Mat& signd) const
{
	Mat thrpos, thrneg;
	compute_thr(imgd, thrpos, thrneg);

	signd = Mat::zeros(imgd.size(), CV_8SC1);

	int numCellcols = imgd.cols / cellSize.width;
	int numCellrows = imgd.rows / cellSize.height;

	Mat cellx;
	Mat celldx;
	Mat posdx;
	posdx = Mat::zeros(cellSize, CV_32FC1);

	for (int i = 0; i < numCellrows; ++i)
	{
		for (int j = 0; j < numCellcols; ++j)
		{
			Rect cellrt(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height);

			celldx = imgd(cellrt);
			cellx = signd(cellrt);
			int thxpos = thrpos.at<int>(i,j);
			int thxneg = thrneg.at<int>(i,j);

			compute_sign(celldx, cellx, thxpos, thxpos);
		}
	}
}

void SLTP3::compute_modemap(const cv::Mat& img, cv::Mat& modemap)const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signx, signy;
	compute_signimg(dx, signx);
	compute_signimg(dy, signy);
	
	modemap = Mat::zeros(dx.size(), CV_8UC1);
	const char* ptrx, *ptry;
	for (int i = 0; i < dx.rows;++i)
	{
		ptrx = signx.ptr<char>(i);
		ptry = signy.ptr<char>(i);
		for (int j = 0; j < dy.cols;++j)
		{		
			int x = ptrx[j];
			int y = ptry[j];
			int mode = signarry[x + 1][y + 1];
			modemap.at<uchar>(i, j) = mode;
		}
	}
}

void SLTP3::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	sltpsvm = _svm;
}


void SLTP3::loadSvmDetector(const string & xmlfile)
{
	sltpsvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void SLTP3::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold/*=0*/, cv::Size winStride/*=cv::Size()*/, const vector<cv::Point>& locations/*=vector<cv::Point>()*/) const
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

	//Í¼ÏñÌî³ä
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

	int numCellcols = paddedimg.cols / cellSize.width;
	int numCellrows = paddedimg.rows / cellSize.height;

	Mat modelmap;
	compute_modemap(paddedimg, modelmap);
	vector<float> featureimg;
	compute_histimg(modelmap, featureimg);

	//Mat winimg;
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
				weights.push_back(response);
			}
		}
	}
	else
	{
		//scan over windows
		//#pragma omp parallel
		{
			Mat winimg;
			for (int j = 0; j < numwinC; ++j)
			{
				for (int i = 0; i < numwinR; ++i)
				{
					Point pt0;
					winimg = paddedimg(Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height));

					int cellindexrow = j*winStride.height / cellSize.height;
					int cellindexcol = i*winStride.width / cellSize.width;

					pt0.x = i*winStride.width - padding.width;
					pt0.y = j*winStride.height - padding.height;;

					vector<float> feature;
					
					get_winfeature(featureimg, feature, Point(cellindexcol, cellindexrow), numCellcols);

					//compute(winimg, feature);

					Mat result;
					//SLTP3svm->predict(winimg,cv::displayStatusBar::)
					float response = sltpsvm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
					response = result.at<float>(0);

					if (response <= -hitThreshold)
					{
						foundlocations.push_back(pt0);
						weights.push_back(-response);
					}

				}
			}
		}
	}
}

void SLTP3::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_SLTP3 :public ParallelLoopBody
{
private:
	const SLTP3* SLTP3;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_SLTP3(const SLTP3* _SLTP3, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		SLTP3 = _SLTP3;
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
		//cout << "SLTP3" << endl;
		for (i = i1; i < i2; i++)
		{
			double scale = levelScale[i];
			Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
			Mat smallerImg(sz, img.type(), smallerImgBuf.data);
			if (sz == img.size())
				smallerImg = Mat(sz, img.type(), img.data, img.step);
			else
				resize(img, smallerImg, sz, 0.0, 0.0, INTER_NEAREST);
			SLTP3->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(SLTP3->winSize.width*scale), cvRound(SLTP3->winSize.height*scale));

			mtx->lock();

			for (size_t j = 0; j < locations.size(); j++)
			{
				//cout << "scale: " << scale << endl;
				/*cout << "locations: " << cvRound(locations[j].x*scale) << " " << cvRound(locations[j].y*scale) << endl;
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

void SLTP3::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels/*=64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
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

	/*vector<Rect> allCandidates;
	vector<double> tempScales;
	vector<double> tempWeights;*/
	vector<double> foundScales;
	foundlocations.clear();
	weights.clear();

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection_SLTP3(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	/*foundScales.clear();
	std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
	foundlocations.clear();
	std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundlocations));
	weights.clear();
	std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(weights));*/

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void SLTP3::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}
