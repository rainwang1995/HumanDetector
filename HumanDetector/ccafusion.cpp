#include "ccafusion.h"
using namespace cv;
void CCAFUSION::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	ccasvm = _svm;
}

void CCAFUSION::loadSvmDetector(const string & xmlfile)
{
	ccasvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void CCAFUSION::loadccamatrix(const string& ymlcca)
{
	FileStorage ccaf(ymlcca, FileStorage::READ);
	
	ccaf["wxcca"] >> wxcca;
	ccaf["wycca"] >> wycca;
	ccaf["wxpca"] >> wxpca;
	ccaf["wypca"] >> wypca;
	fwxcca = arma::fmat((float*)wxcca.data, wxcca.rows, wxcca.cols, false);
	fwycca = arma::fmat((float*)wycca.data, wycca.rows, wycca.cols, false);
	fwxpca = arma::fmat((float*)wxpca.data, wxpca.rows, wxpca.cols, false);
	fwypca = arma::fmat((float*)wypca.data, wypca.rows, wypca.cols, false);

	ftrx = fwxpca*fwxcca;
	ftry = fwypca*fwycca;
	ccaf.release();
	featurelen = wxcca.cols*2;
	flenx = hdddetector.getFeatureLen();
	fleny = pldpkdetector.getFeatureLen();
}

int CCAFUSION::getFeatureLen() const
{
	if (featurelen==0)
	{
		cerr << "NO CCA MATRIX" << endl;
		return -1;
	}
	return featurelen;
}

void CCAFUSION::compute(const cv::Mat& img, vector<float>& features) const
{
	vector<float> featurehdd,featurepldpk;
	hdddetector.compute(img, featurehdd);
	pldpkdetector.compute(img, featurepldpk);
	
	fusionfeature(featurehdd, featurepldpk, features);
}

void CCAFUSION::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (ccasvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}
	Size cellSize = Size(8, 8);
	if (winStride == Size())
	{
		winStride = Size(8,8);
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

	Size cacheStride(gcd(winStride.width, hdddetector.blockStride.width),
		gcd(winStride.height, hdddetector.blockStride.height));
	HDDCache cache(&hdddetector, img, padding, padding, nwindows == 0, cacheStride);
	const HDDCache::BlockData* blockData = &cache.blockData[0];
	int nblocks = cache.nblocks.area();
	int blockHistogramSize = cache.blockHistogramSize;
	vector<float> hddfeature(blockHistogramSize*nblocks);
	vector<float> hddblockHist(blockHistogramSize);

	vector<Mat> integralmap;
	pldpkdetector.compute_integralmaps(paddedimg, integralmap);
	vector<float> pldpkfeature(fleny), feature(featurelen);

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
			float response = ccasvm->predict(feature);
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
			//Mat winimg;
			for (int j = 0,n=0; j < numwinC; ++j)
			{
				for (int i = 0; i < numwinR; ++i,++n)
				{
					Point pt0;
					//winimg = paddedimg(Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height));
					Rect winrt(i*winStride.width, j*winStride.height, winSize.width, winSize.height);

					pt0.x = i*winStride.width - padding.width;
					pt0.y = j*winStride.height - padding.height;

					pt0 = cache.getWindow(paddedImgSize, winStride, (int)n).tl() - Point(padding);
					
					int k;

					for (k = 0; k < nblocks; k++)
					{
						const HDDCache::BlockData& bj = blockData[k];
						Point pt = pt0 + bj.imgOffset;

						const float* vec = cache.getBlock(pt, &hddblockHist[0]);
						memcpy(&hddfeature[blockHistogramSize*k], vec, sizeof(float)*blockHistogramSize);
					}

					//hdddetector.compute(winimg, hddfeature);
					pldpkdetector.compute_histwin(integralmap, winrt, pldpkfeature);
					//double t =(double) getTickCount();
					fusionfeature(hddfeature, pldpkfeature, feature);
					/*double t2 =(double) getTickCount();
					cout << (t2 - t) / getTickFrequency() << endl;*/
					/*vector<float> feature;
					compute(winimg, feature);*/
					Mat result;
					float response = ccasvm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
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

void CCAFUSION::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}


class Parallel_Detection_CCA :public ParallelLoopBody
{
private:
	const CCAFUSION* cca;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_CCA(const CCAFUSION* _cca, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		cca = _cca;
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
			cca->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(cca->winSize.width*scale), cvRound(cca->winSize.height*scale));

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

void CCAFUSION::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (ccasvm->empty())
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

	vector<double> foundScales;
	foundlocations.clear();
	weights.clear();

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection_CCA(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void CCAFUSION::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}

void CCAFUSION::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
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

void CCAFUSION::fusionfeature(vector<float>& f1, vector<float>& f2, vector<float>& f3) const
{
	arma::fmat fax(&f1[0], 1, flenx, false);
	arma::fmat fay(&f2[0], 1, fleny, false);
	//double t = (double)getTickCount();
	arma::fmat fz1 = fax*ftrx;
	arma::fmat fz2 = fay*ftry;
	//arma::fmat fz = fax*ftrx + fay*ftry;
	//double t3 = (double)getTickCount();
	//cout << (t3 - t) / getTickFrequency() << endl;

	f3.resize(featurelen);
	for (int i = 0; i < fz1.n_rows;++i)
	{
		memcpy(&f3[0], fz1.colptr(i), sizeof(float)*fz1.n_cols);
		memcpy(&f3[fz1.n_cols], fz2.colptr(i), sizeof(float)*fz2.n_cols);
	}
	
	/*arma::frowvec f = arma::vectorise(fz, 1);	
	f3 = arma::conv_to<vector<float> >::from(f);*/
}

