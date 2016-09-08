#include "RDSF.h"
#include <algorithm>
#include <numeric>
using namespace cv;

void RDSF::compute(const cv::Mat & img, vector<float>& features) const
{
	//计算binmap
	Mat binmap;
	compute_binmap(img, binmap);
	//计算积分图像
	vector<Mat> integramaps;
	compute_intermap(binmap, integramaps);

	/*for (int i = 0; i < n_bins;++i)
	{
		cout << integramaps[i] << endl;
		imshow("in",integramaps[i]);
		waitKey();
	}*/

	compute_win(integramaps,Rect(0,0,img.cols,img.rows),features);

	integramaps.clear();
}

void RDSF::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	rdsfsvm = _svm;
}

void RDSF::loadSvmDetector(const string& xmlfile)
{
	rdsfsvm = ml::StatModel::load<ml::SVM>(xmlfile);
}


void RDSF::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (rdsfsvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}

	if (winStride == Size())
	{
		winStride = cellSize;
	}

	//图像填充
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

	//计算整幅图binmap
	Mat binmap;
	compute_binmap(paddedimg, binmap);
	vector<Mat> integralmap;
	compute_intermap(binmap, integralmap);

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
			float response = rdsfsvm->predict(feature);
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
			for (int j = 0; j < numwinC; ++j)
			{
				for (int i = 0; i < numwinR; ++i)
				{
					Point pt0;

					Rect winrt(i*winStride.width, j*winStride.height, winSize.width, winSize.height);
				
					pt0.x = i*winStride.width - padding.width;
					pt0.y = j*winStride.height - padding.height;;

					vector<float> feature;
					compute_win(integralmap, winrt, feature);

					//compute(winimg, feature);
					Mat result;
					//sltpsvm->predict(winimg,cv::displayStatusBar::)
					float response = rdsfsvm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
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

void RDSF::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_RDSF :public ParallelLoopBody
{
private:
	const RDSF* rdsf;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_RDSF(const RDSF* _rdsf, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		rdsf = _rdsf;
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
		//cout << "sltp" << endl;
		for (i = i1; i < i2; i++)
		{
			double scale = levelScale[i];
			Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
			Mat smallerImg(sz, img.type(), smallerImgBuf.data);
			if (sz == img.size())
				smallerImg = Mat(sz, img.type(), img.data, img.step);
			else
				resize(img, smallerImg, sz, 0.0, 0.0, INTER_NEAREST);
			rdsf->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(rdsf->winSize.width*scale), cvRound(rdsf->winSize.height*scale));

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

void RDSF::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, 
	double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (rdsfsvm->empty())
	{
		cerr << "svm error" << endl;
		return;
	}
	double scale = 1.;
	int levels = 0;
	vector<double> levelScale;
	scale0 = 1.1;
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
		Parallel_Detection_RDSF(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void RDSF::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, 
	cv::Size winStride /*= cv::Size()*/,
	double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride,
		nlevels, scale0, finalThreshold, usemeanshift);
}



void RDSF::setDefaultParams()
{
	winSize = Size(64, 128);
	cellSize = Size(8, 8);
	n_bins = 30;
}

void RDSF::cal_params()
{
	featurelen = (492 * 491) / 2;
}

void RDSF::compute_binmap(const cv::Mat& img, cv::Mat& binmap) const
{
	float binwidth = n_bins / 10000.0;

	Mat temp;
	img.convertTo(temp, CV_32FC1);
	binmap = temp*binwidth;

	binmap.setTo(29, binmap > 29);
	binmap.convertTo(binmap, CV_8UC1);
}

void RDSF::compute_intermap(const cv::Mat& binmap, vector<cv::Mat>& integmaps) const
{
	integmaps.resize(n_bins);
	for (int i = 0; i < n_bins;++i)
	{
		Mat temp=Mat::zeros(binmap.size(),CV_8UC1);
		temp.setTo(1, binmap == i);

		Mat integralmap;
		integral(temp, integralmap);
		integmaps[i] = integralmap;
	}
}

//void RDSF::compute_feature(const vector<cv::Mat>& integralmaps, vector<float>& features) const
//{
//	//保存所有可能的块
//	//Rect regions[492];
//	vector<Rect> regions(492);
//	for (int i = 1,p=0; i <= 8;++i)
//	{
//		Size upperSize = cellSize*i;
//		int n_rows = (winSize.height - upperSize.height) / cellSize.height+1;
//		int n_cols = (winSize.width - upperSize.width) / cellSize.width+1;
//
//		for (int rowindex = 0; rowindex < n_rows; ++rowindex)
//		{
//			for (int colindex = 0; colindex < n_cols; ++colindex,++p)
//			{
//				Rect currt = Rect(colindex*cellSize.width, rowindex*cellSize.height, upperSize.width, upperSize.height);
//				//当前块
//				regions[p] = currt;	
//			}
//		}
//	}
//
//	features.resize(featurelen);
//	vector<float> histcur(n_bins);
//	vector<float> histnext(n_bins);
//	int p = 0;
//	for (int i = 0; i < 492;++i)
//	{
//		compute_hists(integralmaps, regions[i], histcur);
//		for (int j = i+1; j < 492;++j,++p)
//		{
//			compute_hists(integralmaps, regions[j], histnext);
//			features[p] = compute_similarity(histcur, histnext);
//		}
//	}
//
//	regions.clear();
//	histnext.clear();
//	histnext.clear();
//	//regions.reserve(0);
//}

int RDSF::compute_histvalue(const cv::Mat& integralmap, cv::Rect pos)const
{
//	cout << integralmap.col(0);
	int u = integralmap.at<int>(pos.y + pos.height, pos.x + pos.width );
	int v = integralmap.at<int>(pos.y, pos.x);
	int x = integralmap.at<int>(pos.y + pos.height,pos.x);
	int y = integralmap.at<int>(pos.y , pos.x+pos.width);

	return u + v - x - y;
}

void RDSF::compute_hists(const vector<cv::Mat>& integralmaps, cv::Rect pos, vector<float>& hist) const
{
	hist.resize(n_bins);

	for (int j = 0; j < n_bins; ++j)
	{
		hist[j] = compute_histvalue(integralmaps[j], pos);
	}
	//归一化
	normalise_hist(hist);
}

void RDSF::normalise_hist(vector<float>& hist) const
{
	float sumhist = std::accumulate(hist.begin(), hist.end(), 0.0);

	for (int i = 0; i < hist.size();++i)
	{
		hist[i] /= sumhist;
	}
}

float RDSF::compute_similarity(const vector<float>& hista, const vector<float>& histb) const
{
	CV_Assert(hista.size()==histb.size());
	float s=0.0;
	for (int i = 0; i < hista.size();++i)
	{
		s += (sqrt(hista[i] * histb[i]));
	}
	return s;
}

void RDSF::compute_win(const vector<cv::Mat>& integralmap, cv::Rect roi, vector<float>& features) const
{

	vector<Rect> regions(492);
	for (int i = 1, p = 0; i <= 8; ++i)
	{
		Size upperSize = cellSize*i;
		int n_rows = (winSize.height - upperSize.height) / cellSize.height + 1;
		int n_cols = (winSize.width - upperSize.width) / cellSize.width + 1;

		for (int rowindex = 0; rowindex < n_rows; ++rowindex)
		{
			for (int colindex = 0; colindex < n_cols; ++colindex, ++p)
			{
				Rect currt = Rect(roi.x+colindex*cellSize.width, roi.y+rowindex*cellSize.height, upperSize.width, upperSize.height);
				//当前块
				regions[p] = currt;
			}
		}
	}

	features.resize(featurelen);
	vector<float> histcur(n_bins);
	vector<float> histnext(n_bins);
	int p = 0;
	for (int i = 0; i < 492; ++i)
	{
		compute_hists(integralmap, regions[i], histcur);
		for (int j = i + 1; j < 492; ++j, ++p)
		{
			compute_hists(integralmap, regions[j], histnext);
			features[p] = compute_similarity(histcur, histnext);
		}
	}

	regions.clear();
	histnext.clear();
	histnext.clear();

	//vector<Mat> integralwin(n_bins);
	//Rect borderroi(roi.x, roi.y, roi.width + 1, roi.height + 1);
	////borderroi.x -= 1;
	//for (int i = 0; i < integralmap.size();++i)
	//{
	//	integralwin[i] = integralmap[i](borderroi);
	//}

	//compute_feature(integralwin, features);
}

void RDSF::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
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

