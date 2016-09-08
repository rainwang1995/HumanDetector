#include "SLTPB.h"
using namespace cv;

void SLTPB::compute(const cv::Mat& img, vector<float>& features) const
{
	CV_Assert(img.size() == winSize);

	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signdx, signdy;
	compute_sign(dx, signdx);
	compute_sign(dy, signdy);

	Mat binwin;
	compute_binwin(signdx, signdy, binwin);

	compute_histwin(binwin, features);
}

void SLTPB::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	sltpsvm = _svm;
}

void SLTPB::loadSvmDetector(const string& xmlfile)
{
	sltpsvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void SLTPB::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
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

	int blockperrow = (paddedImgSize.width - blockSize.width) / blockStride.width + 1;
	int blockpercol = (paddedImgSize.height - blockSize.height) / blockStride.height + 1;
	vector<vector<float> > featuresimg(blockperrow*blockpercol);
	vector<bool> cptflags(blockpercol*blockperrow,false);

	Mat patternimg;
	compute_binimg(paddedimg, patternimg);
	//int numCellcols = paddedimg.cols / cellSize.width;
	//int numCellrows = paddedimg.rows / cellSize.height;
	vector<float> featurewin(featurelen);
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
			for (int j = 0; j < numwinC; ++j)
			{
				for (int i = 0; i < numwinR; ++i)
				{
					Point pt0;

					Rect rt=Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height);
					Mat winbin = patternimg(rt);

					int blockindexrow = (j*winStride.height-padding.height) / blockStride.height;
					int blockindexcol = (i*winStride.width-padding.width) / blockStride.width;

					pt0.x = i*winStride.width - padding.width;
					pt0.y = j*winStride.height - padding.height;;

					for (int h = 0,p=0; h < numBlockC;++h)
					{
						for (int k = 0; k < numBlockR;++k,++p)
						{
							Rect blockrect = Rect(k*blockStride.width, h*blockStride.height, blockSize.width, blockSize.height);
							Mat blockroi = winbin(blockrect);
							Point blockpt = Point(blockindexcol+k, blockindexrow+h);

							getblockhist(blockroi, blockpt, &featurewin[p * 4 * 9], featuresimg, cptflags, blockperrow);
						}
					}

					Mat result;
					//sltpsvm->predict(winimg,cv::displayStatusBar::)
					float response = sltpsvm->predict(featurewin, result, ml::StatModel::RAW_OUTPUT);
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

void SLTPB::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_SLTPB :public ParallelLoopBody
{
private:
	const SLTPB* sltp;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_SLTPB(const SLTPB* _sltp, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		sltp = _sltp;
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
			sltp->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(sltp->winSize.width*scale), cvRound(sltp->winSize.height*scale));

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

void SLTPB::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, 
	cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (sltpsvm->empty())
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
		Parallel_Detection_SLTPB(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		//groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void SLTPB::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/,
	cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, 
		scale0, finalThreshold, usemeanshift);
}

void SLTPB::setDefaultParams()
{
	winSize = Size(64, 128);
	blockSize = Size(16, 16);
	cellSize = Size(8, 8);
	blockStride = Size(8, 8);
	signThreshold = 30;

}

void SLTPB::cal_params()
{
	numCellC = winSize.height / cellSize.height;
	numCellR = winSize.width / cellSize.width;

	numCellPerWin = numCellR*numCellC;

	numBlockR = (winSize.width - blockSize.width) / blockStride.width + 1;
	numBlockC = (winSize.height - blockSize.height) / blockStride.height + 1;
	numBlockWin = numBlockR*numBlockC;

	maskdx = (Mat_<char>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
	maskdy = (Mat_<char>(3, 3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);

	featurelen = 9 * 4 * numBlockWin;
	for (int i = 0, p = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j, ++p)
		{
			signarry[i][j] = p;
		}
	}
	compute_gaussian();
	compute_weights();
}

void SLTPB::compute_dxdy(const cv::Mat& img, cv::Mat& dx, cv::Mat& dy) const
{
	filter2D(img, dx, CV_32FC1, maskdx);
	filter2D(img, dy, CV_32FC1, maskdy);
}

void SLTPB::compute_sign(const Mat& dimg, Mat& signimg) const
{
	signimg = Mat::zeros(dimg.size(), CV_8SC1);

	signimg.setTo(-1, dimg <= -signThreshold);
	signimg.setTo(1, dimg >= signThreshold);
}

void SLTPB::compute_binwin(const cv::Mat & signimgx, const cv::Mat & signimgy, cv::Mat & patterns) const
{
	patterns = Mat::zeros(signimgx.size(), CV_8UC1);
	const char* ptrx, *ptry;
	char* ptrp;
	for (int i = 0; i < signimgx.rows; ++i)
	{
		ptrx = signimgx.ptr<char>(i);
		ptry = signimgy.ptr<char>(i);
		ptrp = patterns.ptr<char>(i);
		for (int j = 0; j < signimgx.cols; ++j)
		{
			int x = ptrx[j];
			int y = ptry[j];
			int mode = signarry[x + 1][y + 1];
			ptrp[j] = mode;
		}
	}
}

void SLTPB::compute_binimg(const cv::Mat& img, cv::Mat& patterns) const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signdx, signdy;
	compute_sign(dx, signdx);
	compute_sign(dy, signdy);
	compute_binwin(signdx, signdy, patterns);
}

void SLTPB::getblockhist(const cv::Mat& blockimg, Point pt, float* blockhist, vector<vector<float> >& imagehist, vector<bool> flags,int blockperrow) const
{
	int index = pt.y*blockperrow + pt.x;
	if (flags[index])
	{
		memcpy(blockhist, &imagehist[index][0], sizeof(float) * 4 * 9);
		return;
	}

	compute_histblock(blockimg, blockhist);
	imagehist[index].resize(4 * 9);
	memcpy(&imagehist[index][0], blockhist, sizeof(float) * 4 * 9);
	flags[index] = true;	
}

void SLTPB::compute_gaussian()
{
	gaussianweights = Mat_<float>::zeros(blockSize);
	float sigma = (blockSize.width + blockSize.height) / 8.;
	float scale = 1.f / (sigma*sigma * 2);
	int i, j;
	for (i = 0; i < blockSize.height; i++)
		for (j = 0; j < blockSize.width; j++)
		{
			float di = i - blockSize.height*0.5f;
			float dj = j - blockSize.width*0.5f;
			gaussianweights(i, j) = std::exp(-(di*di + dj*dj)*scale);
		}
}

void SLTPB::compute_weights()
{
	Size ncells(blockSize.width / cellSize.width, blockSize.height / cellSize.height);
	blockweights = Mat::zeros(blockSize, CV_32FC4);
	blockOfs = Mat::zeros(blockSize, CV_32SC4);
	int nbins = 9;
	for (int i = 0; i < blockSize.width; ++i)
	{
		for (int j = 0; j < blockSize.height; ++j)
		{
			float cellX = (i + 0.5f) / cellSize.width - 0.5;
			float cellY = (j + 0.5) / cellSize.height - 0.5;

			int icellX0 = cvFloor(cellX);
			int icellY0 = cvFloor(cellY);
			int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
			cellX -= icellX0;
			cellY -= icellY0;

			if ((unsigned)icellX0 < (unsigned)ncells.width &&
				(unsigned)icellX1 < (unsigned)ncells.width)
			{
				if ((unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height)
				{
					//
					blockweights.at<Vec4f>(j, i)[0] = (1.f - cellX)*(1.f - cellY);
					blockweights.at<Vec4f>(j, i)[1] = cellX*(1.f - cellY);
					blockweights.at<Vec4f>(j, i)[2] = (1.f - cellX)*cellY;
					blockweights.at<Vec4f>(j, i)[3] = cellX*cellY;

					blockOfs.at<Vec4i>(j, i)[0] = (icellX0*ncells.height + icellY0)*nbins;
					blockOfs.at<Vec4i>(j, i)[1] = (icellX1*ncells.height + icellY0)*nbins;
					blockOfs.at<Vec4i>(j, i)[2] = (icellX0*ncells.height + icellY1)*nbins;
					blockOfs.at<Vec4i>(j, i)[3] = (icellX1*ncells.height + icellY1)*nbins;
				}
				else
				{
					if ((unsigned)icellY0 < (unsigned)ncells.height)
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;
					}
					blockweights.at<Vec4f>(j, i)[0] = (1.f - cellX)*cellY;
					blockweights.at<Vec4f>(j, i)[1] = cellX*cellY;

					blockOfs.at<Vec4i>(j, i)[0] = (icellX0*ncells.height + icellY1)*nbins;
					blockOfs.at<Vec4i>(j, i)[1] = (icellX1*ncells.height + icellY1)*nbins;

				}
			}

			else
			{
				if ((unsigned)icellX0 < (unsigned)ncells.width)
				{
					icellX1 = icellX0;
					cellX = 1.f - cellX;
				}

				if ((unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height)
				{
					blockweights.at<Vec4f>(j, i)[0] = cellX*(1.f - cellY);
					blockweights.at<Vec4f>(j, i)[1] = cellX*cellY;

					blockOfs.at<Vec4i>(j, i)[0] = (icellX1*ncells.height + icellY0)*nbins;
					blockOfs.at<Vec4i>(j, i)[1] = (icellX1*ncells.height + icellY1)*nbins;
				}
				else
				{
					if ((unsigned)icellY0 < (unsigned)ncells.height)
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;
					}
					blockweights.at<Vec4f>(j, i)[0] = cellX*cellY;
					blockOfs.at<Vec4i>(j, i)[0] = (icellX1*ncells.height + icellY1)*nbins;
				}
			}
		}
	}
}

void SLTPB::get_winfeature(vector<float>& featuresimg, vector<float>& featureswin, cv::Point& startpos, int blocksperrow) const
{
	featureswin.clear();
	featureswin.resize(featurelen,0);

	int featurelenperrow = 4*9*blocksperrow;
	for (int i = 0; i < numBlockC; ++i)
	{
		int sindex = featurelenperrow*(startpos.y + i) + startpos.x;
		sindex *= (9*4);
		memcpy(&featureswin[i*featurelenperrow], &featuresimg[sindex], sizeof(float)*featurelenperrow);
	}
}

void SLTPB::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
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

void SLTPB::compute_img(const cv::Mat& img, vector<float>& features) const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signdx, signdy;
	compute_sign(dx, signdx);
	compute_sign(dy, signdy);

	Mat binwin;
	compute_binwin(signdx, signdy, binwin);
	compute_histimage(binwin, features);
}

void SLTPB::normalizeBlockHistogram(float* blockhist) const
{
	float* hist = &blockhist[0];
	size_t i, sz = 4*9;

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

void SLTPB::compute_histblock(const cv::Mat& binblock, float* hist)const
{
	for (int i = 0; i < binblock.rows;++i)
	{
		for (int j = 0; j < binblock.cols;++j)
		{
			for (int k = 0; k < 4;++k)
			{
				float w = blockweights.at<Vec4f>(i,j)[k];
				if (w!=0)
				{
					w = w*gaussianweights.at<float>(i,j);
					int cellOfs = blockOfs.at<Vec4i>(i,j)[k];
					int binindex = binblock.at<uchar>(i, j);
					hist[cellOfs + binindex] += w;
				}
			}
		}
	}

	normalizeBlockHistogram(hist);
}

void SLTPB::compute_histwin(const cv::Mat& win, vector<float>& feature) const
{
	feature.clear();

	feature.resize(4 * 9 * numBlockWin);

	for (int i = 0, p = 0; i < numBlockC; ++i)
	{
		for (int j = 0; j < numBlockR; ++j, ++p)
		{
			Rect rt(j*blockStride.width, i*blockStride.height, blockSize.width, blockSize.height);
			Mat blockbin = win(rt);

			compute_histblock(blockbin, &feature[4 * 9 * p]);
		}
	}
}

void SLTPB::compute_histimage(const cv::Mat& binimg, vector<float> hist) const
{
	hist.clear();
	int nblocksR = (binimg.cols - blockSize.width) / blockStride.width + 1;
	int nblocksC = (binimg.rows - blockSize.height) / blockStride.height + 1;
	hist.resize(4*9*nblocksR*nblocksC);

	for (int i = 0,p=0; i < nblocksC;++i)
	{
		for (int j = 0; j < nblocksR;++j,++p)
		{
			Rect rt(j*blockStride.width, i*blockStride.height, blockSize.width, blockSize.height);
			Mat blockbin = binimg(rt);

			compute_histblock(blockbin, &hist[4 * 9 * p]);
		}
	}
}
