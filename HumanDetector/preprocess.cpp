#include "preprocess.h"
#include <omp.h>
#include "our_fmm.hpp"
//using namespace cv;


deque<cv::Mat> preProcessing::framequeue=deque<cv::Mat>();

void preProcessing::pixelFilter(cv::Mat & src, cv::Mat& dst2)
{
	cv::Mat dst = src.clone();
	
//
	int width = src.cols;
	int height = src.rows;

	int widthBound = width - 1;
	int heightBound = height - 1;
	dst2 = cv::Mat::zeros(src.size(), CV_16UC1);
	//dst2 = dst;
	int **filterCollection = new int *[24];
	for (int i = 0; i < 24; ++i)
	{
		filterCollection[i] = new int[2];
	}

	ushort *srcdata = (ushort*)dst.data;
	ushort *dstdata = (ushort*)dst2.data;
	int widthstep = dst.step1();
//#pragma omp parallel for
	for (int rowindex = 0; rowindex < height; ++rowindex)
	{
		for (int colindex = 0; colindex < width; ++colindex)
		{
			int depthindex = colindex + rowindex*widthstep;
			if (srcdata[depthindex] == 0)
			{
				for (int i = 0; i < 24; ++i)
				{
					for (int j = 0; j < 2; ++j)
					{
						filterCollection[i][j] = 0;
					}
				}

				int innerBandCount = 0;
				int outerBandCount = 0;

				for (int yi = -2; yi < 3; ++yi)
				{
					for (int xi = -2; xi < 3; ++xi)
					{
						if (xi != 0 || yi != 0)
						{
							int xSearch = colindex + xi;
							int ySearch = rowindex + yi;

							if (xSearch >= 0 && xSearch <= widthBound&&ySearch >= 0 && ySearch <= heightBound)
							{
								int searchindex = xSearch + ySearch*widthstep;
								uchar *rowdata = src.ptr<uchar>(ySearch);
								if (srcdata[searchindex] != 0)
								{
									for (int i = 0; i < 24; ++i)
									{

										if (filterCollection[i][0] == srcdata[searchindex])
										{
											++filterCollection[i][1];
											break;
										}
										else if (filterCollection[i][0] == 0)
										{
											filterCollection[i][0] = srcdata[searchindex];
											++filterCollection[i][1];
											break;
										}
									}

									if (yi != 2 && yi != -2 && xi != 2 && xi != -2)
									{
										++innerBandCount;
									}
									else
										++outerBandCount;
								}
							}
						}
					}
				}


				if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold)
				{
					int frequency = 0;
					int depth = 0;
					for (int i = 0; i < 24; ++i)
					{
						if (filterCollection[i][0] == 0)
						{
							break;
						}
						if (filterCollection[i][1]>frequency)
						{
							depth = filterCollection[i][0];
							frequency = filterCollection[i][1];
						}
					}
					dstdata[depthindex] = depth;
				}
			}
			else
				dstdata[depthindex] = srcdata[depthindex];
		}
	}
//
//	dst.setTo(8000, src > 8000);
	for (int i = 0; i < 24; ++i)
	{
		delete[] filterCollection[i];
	}
	delete[] filterCollection;

	//dst2 = dst;
	//medianBlur(dst2, dst2, 3);

	//double minv, maxv;

	//minMaxLoc(dst2, &minv, &maxv);
	//dst2.setTo((ushort)maxv, dst2 == 0);
	/*if (framequeue.size() == 4)
	{
		framequeue.pop_front();
	}
	framequeue.push_back(dst);*/

}

void preProcessing::contextFilter(cv::Mat & src, cv::Mat & dst)
{
	dst = src;
	if (framequeue.size()==0)
	{
		return;
	}
	ushort* srcdata = (ushort*)dst.data;
	int widthstep = dst.step1();
	for (int rowindex = 0; rowindex < dst.rows;++rowindex)
	{
		for (int colindex = 0; colindex < dst.cols;++colindex)
		{
			int index = colindex + rowindex*widthstep;
			if (srcdata[index]==0)
			{
				for (int i = framequeue.size()-1; i >=0;--i)
				{
					ushort* tempptr = (ushort*)framequeue[i].data;
					ushort temp = tempptr[index];
					if (temp!=0)
					{
						srcdata[index] = temp;
					}
				}
			}
		}
	}	
}

void preProcessing::inpaintrawdepth(cv::Mat& src, cv::Mat& dst, double inpaintRange, float alpha)
{
	inpaint(src, src==0,dst, inpaintRange, alpha);
}

//近邻补洞
void filterfunc1(cv::Mat& src, cv::Mat& dst)
{
	preProcessing::pixelFilter(src, dst);
}

//用最大值填充
void filterfunc2(cv::Mat& src, cv::Mat& dst)
{
	dst = src.clone();
	double minv, maxv;

	minMaxLoc(dst, &minv, &maxv);
	dst.setTo((ushort)maxv, dst == 0);
	medianBlur(dst, dst, 3);
}

//inpaint
void filterfunc3(cv::Mat& src, cv::Mat& dst)
{
	preProcessing::inpaintrawdepth(src, dst, 3, 0.5);
}

// 最近邻填充后，最大值填充
void filterfunc4(cv::Mat& src, cv::Mat& dst)
{
	preProcessing::pixelFilter(src, dst);
	double minv, maxv;

	minMaxLoc(dst, &minv, &maxv);
	dst.setTo((ushort)maxv, dst == 0);
}

//