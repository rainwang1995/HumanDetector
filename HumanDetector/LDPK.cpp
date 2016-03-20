#include "LDPK.h"
#include <bitset>
#include <algorithm>
#include <functional>
using namespace cv;
using namespace std;

void LDPK::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	ldpksvm = _svm;
}

void LDPK::loadSvmDetector(const string& xmlfile)
{
	ldpksvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void LDPK::compute(const cv::Mat& img, vector<float>& features) const
{
	Mat ltpimg;
	compute_Ltpvalue(img, ltpimg);
	compute_histwin(ltpimg, features);
}

void LDPK::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (ldpksvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}

	if (winStride == Size())
	{
		winStride = Size(8,8);
	}

	//Í¼ÏñÌî³ä
	Size stride(gcd(winStride.width, blockSize.width), gcd(winStride.height, blockSize.height));

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


	if (winStride.width > blockSize.width || winStride.height > blockSize.height)
	{
		stride = blockSize;
	}
	else
		stride = winStride;

	int blockperrow = (paddedImgSize.width - blockSize.width) / stride.width + 1;
	int blockpercol = (paddedImgSize.height - blockSize.height) / stride.height + 1;
	vector<vector<float> > featuresimg(blockperrow*blockpercol);
	vector<bool> cptflags(blockpercol*blockperrow, false);

	Mat patternimg;
	compute_Ltpvalue(paddedimg, patternimg);
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

					Rect rt = Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height);
					Mat winbin = patternimg(rt);

					int blockindexrow = (j*winStride.height - padding.height) / stride.height;
					int blockindexcol = (i*winStride.width - padding.width) / stride.width;

					pt0.x = i*winStride.width - padding.width;
					pt0.y = j*winStride.height - padding.height;;

					for (int j = 0, p = 0; j < numBlockC; ++j)
					{
						for (int k = 0; k < numBlockR; ++k, ++p)
						{
							Rect blockrect = Rect(k*blockSize.width, j*blockSize.height, blockSize.width, blockSize.height);
							Mat blockroi = winbin(blockrect);
							Point blockpt = Point(blockindexrow, blockindexcol);

							getblockhist(blockroi, blockpt, &featurewin[p * 4 * 9], featuresimg, cptflags, blockperrow);
						}
					}

					Mat result;
					//sltpsvm->predict(winimg,cv::displayStatusBar::)
					float response = ldpksvm->predict(featurewin, result, ml::StatModel::RAW_OUTPUT);
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

void LDPK::setDefaultParams()
{
	winSize = Size(64, 128);
	blockSize = Size(16, 16);
	K = 3;
}

void LDPK::cal_params()
{
	numBlockC = winSize.height / blockSize.height;
	numBlockR = winSize.width / blockSize.width;

	numBlockPerWin = numBlockR*numBlockC;
	featurelen = featurelenblock*numBlockPerWin;

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
	lookUpTable=Mat::zeros(1, 256, CV_8U);

	for (int i = 7,p=0; i >=2 ;--i)
	{
		for (int j = i - 1; j >= 1;--j)
		{
			for (int k = j - 1; k >= 0;--k,++p)
			{
				b.reset();
				b.set(i);
				b.set(j);
				b.set(k);

				uchar v = (uchar)b.to_ulong();
				lookUpTable.at<uchar>(p) = v;
			}
		}
	}
}

void LDPK::compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimg) const
{
	ltpimg = Mat::zeros(src.size(), CV_8UC1);

	Mat dismap[8];
	for (int i = 0; i < 8; ++i)
	{
		filter2D(src, dismap[i], CV_32FC1, masks[i]);
	}

	for (int i = 0; i < src.rows;++i)
	{
		for (int j = 0; j < src.cols;++j)
		{
			vector<float> kv(8);
			for (int k = 0; k < 8;++k)
			{
				kv[k] = dismap[k].at<float>(i, j);
			}
			sort(kv.begin(), kv.end(),greater<float>());

			int kth = kv[K];
			uchar sumv = 0;
			for (int k = 0; k < 8;++k)
			{
				if (dismap[k].at<float>(i, j) >= kth)
				{
					sumv += ((1 << (7 - k)) * 1);
				}
				else
					sumv += 0;
			}
			ltpimg.at<uchar>(i, j) = sumv;
		}
	}

	LUT(ltpimg, lookUpTable, ltpimg);
}

void LDPK::compute_histblock(const cv::Mat& blockltpimg, float* feature) const
{
	memset(feature, 0, sizeof(float)*featurelenblock);
	for (int i = 0; i < blockltpimg.rows;++i)
	{
		for (int j = 0; j < blockltpimg.cols;++j)
		{
			++feature[blockltpimg.at<uchar>(i, j)];
		}
	}
}

void LDPK::compute_histwin(const cv::Mat& ltpimg, vector<float>& features) const
{
	features.clear();
	features.resize(featurelen, 0);

	Mat ltpblock;
	//#pragma omp parallel for
	for (int i = 0,p=0; i < numBlockC; ++i)
	{
		for (int j = 0; j < numBlockR; ++j,++p)
		{
			//int p = i*numBlockR + j;
			Rect blockroi(j*blockSize.width, i*blockSize.height, blockSize.width, blockSize.height);

			ltpblock = ltpimg(blockroi);
			compute_histblock(ltpblock, &features[p*featurelenblock]);
		}
	}
}

//no-implemention
void LDPK::compute_histimg(const cv::Mat& ltpimg, vector<float>& features,Size winStride) const
{
	Size stride;
	if (winStride.width > blockSize.width||winStride.height>blockSize.height)
	{
		stride = blockSize;
	}
	else
		stride = winStride;
	int nblocksperrow = (ltpimg.cols - blockSize.width) / stride.width + 1;
	int nblockspercol = (ltpimg.rows - blockSize.height) / stride.height + 1;

	features.clear();
	features.resize(nblockspercol*nblocksperrow*featurelenblock);

	for (int i = 0,p=0; i < nblockspercol;++i)
	{
		for (int j = 0; j < nblocksperrow;++j,++p)
		{
			Rect rt = Rect(i*stride.width, j*stride.height, blockSize.width, blockSize.height);
			Mat blockimg = ltpimg(rt);
			compute_histblock(blockimg, &features[p*featurelenblock]);
		}
	}
}

void LDPK::getblockhist(const cv::Mat& blockimg, cv::Point pt, float* blockhist, vector<vector<float> >& imagehist, vector<bool> flags, int blockperrow) const
{
	int index = pt.y*blockperrow + pt.x;
	if (flags[index])
	{
		memcpy(blockhist, &imagehist[index], sizeof(float) * 4 * 9);
		return;
	}

	compute_histblock(blockimg, blockhist);
	imagehist[index].resize(4 * 9);
	memcpy(&imagehist[index][0], blockhist, sizeof(float) * 4 * 9);
	flags[index] = true;
}

