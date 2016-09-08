#include "PLDPK.h"
#include <bitset>

using namespace cv;
void PLDPK::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	ldpksvm = _svm;
}

void PLDPK::setDefaultParams()
{
	winSize = Size(64, 128);
	K = 3;
	plevels = 3;
}

void PLDPK::cal_params()
{
	blockSize = Size(16, 16);
	blockrows = winSize.height / blockSize.height;
	blockcols = winSize.width / blockSize.width;
	blocksinlevels = pow(4, plevels) / (4 - 1);
	/*for (int i = 0; i < blocksinlevels;++i)
	{
		blocksinlevels +=((pow(2, i) - 1)*(pow(2, i) - 1));
	}*/
	featurelen = featurelenblock*(blocksinlevels + blockrows*blockcols);

	masks.resize(8);
	masks[0] = (Mat_<char>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
	masks[1] = (Mat_<char>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
	masks[2] = (Mat_<char>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
	masks[3] = (Mat_<char>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
	masks[4] = (Mat_<char>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
	masks[5] = (Mat_<char>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
	masks[6] = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
	masks[7] = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);

	bitset<8> b;
	lookUpTable = Mat::zeros(1, 256, CV_8U);

	for (int i = 7, p = 0; i >= 2; --i)
	{
		for (int j = i - 1; j >= 1; --j)
		{
			for (int k = j - 1; k >= 0; --k, ++p)
			{
				b.reset();
				b.set(i);
				b.set(j);
				b.set(k);

				uchar v = (uchar)b.to_ulong();
				lookUpTable.at<uchar>(v) = p;
			}
		}
	}
}

void PLDPK::compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimg) const
{
	ltpimg = Mat::zeros(src.size(), CV_8UC1);

	Mat dismap[8];
	for (int i = 0; i < 8; ++i)
	{
		filter2D(src, dismap[i], CV_32FC1, masks[i]);
	}

	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			vector<float> kv(8);
			for (int k = 0; k < 8; ++k)
			{
				kv[k] = dismap[k].at<float>(i, j);
			}

			int sortindex[8] = { 0,1,2,3,4,5,6,7 };

			//插入排序
			for (int k = 1; k < 8; ++k)
			{
				int h = k - 1;
				float temp = kv[k];
				int tempsort = sortindex[k];
				while (h >= 0 && kv[h] < temp)
				{
					kv[h + 1] = kv[h];
					sortindex[h + 1] = sortindex[h];
					--h;
				}
				kv[h + 1] = temp;
				sortindex[h + 1] = tempsort;
			}

			vector<int> v(8, 0);
			for (int k = 0; k < K; ++k)
			{
				v[sortindex[k]] = 1;
			}
			//sort(kv.begin(), kv.end(),greater<float>());

			uchar sumv = 0;
			for (int k = 0; k < 8; ++k)
			{
				sumv += ((1 << (7 - k)) * v[k]);
			}
			ltpimg.at<uchar>(i, j) = sumv;
		}
	}

	LUT(ltpimg, lookUpTable, ltpimg);
}

//计算积分图
void PLDPK::compute_integralmap(const cv::Mat& ltpimg, vector<cv::Mat>& integralmaps) const
{
	//integralmaps.clear();
	//integralmaps.resize(featurelenblock,Mat::zeros(ltpimg.rows+1,ltpimg.cols+1,CV_32SC1));
	integralmaps.resize(featurelenblock);

	//cout << ltpimg << endl;
	/*for (int i = 0; i < ltpimg.rows;++i)
	{
		for (int j = 0; j < ltpimg.cols;++j)
		{
			int val = ltpimg.at<uchar>(i, j);
			integralmaps[val].at<int>(i + 1, j + 1) = integralmaps[val].at<int>(i + 1, j) + integralmaps[val].at<int>(i, j + 1)
				- integralmaps[val].at<int>(i, j) + 1;

			for (int k = 0; k < featurelenblock;++k)
			{
				if (k!=val)
				{
					integralmaps[k].at<int>(i + 1, j + 1) = integralmaps[k].at<int>(i + 1, j) + integralmaps[val].at<int>(i, j + 1)
						- integralmaps[k].at<int>(i, j) +0;
				}
			}
		}
	}*/
	//double t = (double)getTickCount();

	for (int i = 0; i < featurelenblock;++i)
	{
		Mat temp = Mat::zeros(ltpimg.size(), CV_8UC1);
		temp.setTo(1, ltpimg == i);

		Mat intergralimg;
		integral(temp, intergralimg);
		integralmaps[i] = intergralimg;
	}
	//cout << ((double)getTickCount() - t) / getTickFrequency() << endl;

}

int PLDPK::compute_histvalue(const cv::Mat& integralmap, cv::Rect pos) const
{
	int u = integralmap.at<int>(pos.y + pos.height, pos.x + pos.width);
	int v = integralmap.at<int>(pos.y, pos.x);
	int x = integralmap.at<int>(pos.y + pos.height, pos.x);
	int y = integralmap.at<int>(pos.y, pos.x + pos.width);

	return u + v - x - y;
}

void PLDPK::compute_hists(const vector<cv::Mat>& integralmaps, cv::Rect pos, float* hist) const
{
	
	for (int j = 0; j < featurelenblock; ++j)
	{
		hist[j] = compute_histvalue(integralmaps[j], pos);
	}
	//归一化
	normalise_hist(hist);
}

void PLDPK::normalise_hist(float* blockhist) const
{
	float* hist = &blockhist[0];
	size_t i, sz = featurelenblock;

	//另一种归一化
	/*float sum = std::accumulate(blockhist, blockhist + featurelenblock, 0.0);
	for (int i = 0; i < featurelenblock;++i)
	{
	hist[i] /= sum;
	}*/

	float sum = 0;
	for (i = 0; i < sz; i++)
		sum += hist[i] * hist[i];

	float scale = 1.f / (std::sqrt(sum) + sz*0.1f), thresh = 0.8;

	for (i = 0, sum = 0; i < sz; i++)
	{
		hist[i] = std::min(hist[i] * scale, thresh);
		sum += hist[i] * hist[i];
	}

	scale = 1.f / (std::sqrt(sum) + 1e-3f);

	for (i = 0; i < sz; i++)
		hist[i] *= scale;
}

//void PLDPK::compute_feature(const vector<cv::Mat>& integralmaps, vector<float>& features) const
//{
//	features.resize(featurelen);
//
//	for (int l = 0,p=0; l < plevels;++l)
//	{
//		int blocksperrow = pow(2, 1);
//		int blockspercol = pow(2, 1);
//
//		int blockWidth = winSize.width / blocksperrow;
//		int blockHeight = winSize.height / blockspercol;
//
//		for (int i = 0; i < blockspercol;++i)
//		{
//			for (int j = 0; j < blocksperrow;++j,++p)
//			{
//				Rect roi = Rect(j*blockWidth, i*blockHeight, blockWidth, blockHeight);
//
//				compute_hists(integralmaps, roi, &features[p*featurelenblock]);
//			}
//		}		
//	}
//
//}

void PLDPK::compute_histwin(const vector<cv::Mat>& integralmap, cv::Rect roi, vector<float>& features) const
{

	features.resize(featurelen);

	for (int l = 0, p = 0; l < plevels; ++l)
	{
		int blocksperrow = pow(2, 1);
		int blockspercol = pow(2, 1);

		int blockWidth = winSize.width / blocksperrow;
		int blockHeight = winSize.height / blockspercol;

		float w = 1 / (pow(2, plevels - l));

		for (int i = 0; i < blockspercol; ++i)
		{
			for (int j = 0; j < blocksperrow; ++j, ++p)
			{
				Rect roiblock = Rect(roi.x+j*blockWidth, roi.y+i*blockHeight, blockWidth, blockHeight);

				compute_hists(integralmap, roiblock, &features[p*featurelenblock]);
				for (int k = 0; k < featurelenblock;++k)
				{
					features[p*featurelenblock + k] *= w;
				}
			}
		}

		/*for (int i = 0; i < blockspercol-1; ++i)
		{
			for (int j = 0; j < blocksperrow-1; ++j, ++p)
			{
				Rect roiblock = Rect(roi.x+blockWidth/2 + j*blockWidth, roi.y + i*blockHeight+blockHeight/2, blockWidth, blockHeight);

				compute_hists(integralmap, roiblock, &features[p*featurelenblock]);
				for (int k = 0; k < featurelenblock; ++k)
				{
					features[p*featurelenblock + k] *= w;
				}
			}
		}*/
	}

	//add blockSize(16,16)
	float* fptr = &features[blocksinlevels*featurelenblock];
	for (int i = 0,p=0; i < blockrows;++i)
	{
		for (int j = 0; j < blockcols;++j,++p)
		{
			Rect roiblock = Rect(roi.x + j*blockSize.width, roi.y + i*blockSize.height, blockSize.width, blockSize.height);

			compute_hists(integralmap, roiblock, &fptr[p*featurelenblock]);
		}
	}

	//vector<Mat> integralwin(featurelenblock);
	//Rect borderroi(roi.x, roi.y, roi.width + 1, roi.height + 1);
	////borderroi.x -= 1;
	//for (int i = 0; i < integralmap.size(); ++i)
	//{
	//	integralwin[i] = integralmap[i](borderroi);
	//}

	//compute_feature(integralwin, features);
}

void PLDPK::compute_integralmaps(const cv::Mat& src, vector<cv::Mat>& integralmaps) const
{
	Mat ltpimg;
	compute_Ltpvalue(src, ltpimg);
	compute_integralmap(ltpimg, integralmaps);
}

void PLDPK::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
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

void PLDPK::loadSvmDetector(const string& xmlfile)
{
	ldpksvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void PLDPK::compute(const cv::Mat& img, vector<float>& features) const
{
	vector<Mat> integralmaps;
	compute_integralmaps(img, integralmaps);
	compute_histwin(integralmaps, Rect(0, 0, img.cols, img.rows), features);
}

void PLDPK::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (ldpksvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}
	
	Size cellSize(8, 8);
	if (winStride == Size())
	{
		winStride = Size(8,8);
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

	vector<Mat> integralmap;
	compute_integralmaps(paddedimg, integralmap);

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
			float response = ldpksvm->predict(feature);
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
					compute_histwin(integralmap, winrt, feature);

					//compute(winimg, feature);
					Mat result;
					//sltpsvm->predict(winimg,cv::displayStatusBar::)
					float response = ldpksvm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
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

void PLDPK::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_PLDPK :public ParallelLoopBody
{
private:
	const PLDPK* ldpk;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_PLDPK(const PLDPK* _ldpk, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		ldpk = _ldpk;
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
			ldpk->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(ldpk->winSize.width*scale), cvRound(ldpk->winSize.height*scale));

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
void PLDPK::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (ldpksvm->empty())
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

	/*vector<Rect> allCandidates;
	vector<double> tempScales;
	vector<double> tempWeights;*/
	vector<double> foundScales;
	foundlocations.clear();
	weights.clear();

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection_PLDPK(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		//groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void PLDPK::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}

