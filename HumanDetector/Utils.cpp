#include "Utils.h"
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>

#define EPSILON 0.0000001

void Utils::Histogram::getHist(Mat & src, Mat & depth_hist, const int histSize,const int dims, const int channels)
{
	float range[] = { 0,8000 };
	const float* histRange[] = { range };
	calcHist(&src, 1, &channels,Mat(),depth_hist,dims,&histSize,histRange);
	depth_hist.setTo(0, depth_hist < 20);
}

void Utils::Histogram::plotHist(Mat & hist, Mat & showimg)
{
	if (hist.channels() != 1) return;
	double maxVal = 0;
	double minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal);
	int histSize = hist.rows;
	showimg.create(histSize, histSize, CV_8U);
	showimg.setTo(Scalar(255));

	int hpt = (int)(0.9*histSize);
	for (int h = 0; h < histSize;++h)
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(showimg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
	}
}

void Utils::Histogram::getMaxPeaks(Mat & hist, vector<pair<int, int>>& peaks)
{
	int histSize = hist.rows;
	if (histSize==0)
	{
		return;
	}
	if (histSize==1)
	{
		peaks.push_back(make_pair(hist.at<float>(0), 0));
	}
	else
	{
		for (size_t i = 0; i < histSize;++i)
		{
			if (i>0&&i<histSize-1)
			{
				if (hist.at<float>(i - 1) < hist.at<float>(i) && hist.at<float>(i) > hist.at<float>(i + 1))
				{
					peaks.push_back(make_pair(hist.at<float>(i), i));
				}
			}
			else if (i == 0)
			{
				/*if (hist.at<float>(i)>hist.at<float>(i + 1))
				{
					peaks.push_back(make_pair(hist.at<float>(i), i));
				}*/
				continue;
			}
			else
			{
				if (hist.at<float>(i - 1) < hist.at<float>(i))
					{
						peaks.push_back(make_pair(hist.at<float>(i), i));
					}
			}
		}
	}
}

void Utils::Histogram::markMaxPeaks(Mat & hist, Mat & histimg)
{
	if (hist.channels() != 1) return;
	double maxVal = 0;
	double minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal);
	int histSize = hist.rows;
	histimg.create(histSize, histSize, CV_8UC3);
	histimg.setTo(Scalar(255,255,255));

	int hpt = (int)(0.9*histSize);
	for (int h = 0; h < histSize; ++h)
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);

		if (h > 0 && h<histSize - 1)
		{
			if (hist.at<float>(h - 1) < binVal && binVal > hist.at<float>(h + 1))
			{
				line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar(255, 0, 0));
			}
			else
				line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));

		}
		else if (h == 0&&h<histSize-1)
		{
			if (binVal>hist.at<float>(h + 1))
			{
				line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar(255, 0, 0));
			}
			else
				line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
		}
		else if (h==histSize-1&&h>0)
		{
			if (hist.at<float>(h - 1) < binVal)
			{
				line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar(255, 0, 0));
			}
			else
				line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
		}
		else
			line(histimg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
	}
	/*vector<pair<int, int> >peaks;
	getMaxPeaks(hist, peaks);
	for (int i = 0; i < peaks.size();++i)
	{
		int binVal = peaks[i].first;
		int pos = peaks[i].second;
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histimg, Point(pos, histSize), Point(pos, histSize - intensity), Scalar(255,0,0));
	}*/
}

void Utils::MeanShift::cluster(const vector<Point2f>& points, vector<Point2f>& centers, vector<int>& labels,double kernel_bandwidth/*=2*/)
{
	vector<bool> stop_moving(points.size(),false);
	stop_moving.reserve(points.size());

	labels = vector<int>(points.size(), -1);

	vector<Point2f> shifted_points = points;
	double max_shift_distance = 1e30;
	while (max_shift_distance>EPSILON)
	{
		max_shift_distance = 0;
		for (int i = 0; i < shifted_points.size();++i)
		{
			if (!stop_moving[i])
			{
				Point2f point_new = shift_point(shifted_points[i], points, kernel_bandwidth);
				double shift_distance = Utils::euclidean_distance(point_new, shifted_points[i]);
				if (shift_distance>max_shift_distance)
				{
					max_shift_distance = shift_distance;
				}
				if (shift_distance<EPSILON)
				{
					stop_moving[i] = true;
				}

				shifted_points[i] = point_new;
			}
		}
	}

	stop_moving.clear();
	
	int clusters = 0;

	for (int i = 0; i < shifted_points.size();++i)
	{
		if (labels[i]==-1)
		{	
			++clusters;

			centers.push_back(shifted_points[i]);
			for (int j = i; j < shifted_points.size();++j)
			{
				if (shifted_points[i]==shifted_points[j])
				{
					labels[j] = clusters;
				}
			}
		}
	}
}

void Utils::MeanShift::set_kernel(double(*_kernel_func)(double, double))
{
	if (!_kernel_func) {
		kernel_func = gaussian_kernel;
	}
	else {
		kernel_func = _kernel_func;
	}
}

Point2f Utils::MeanShift::shift_point(const Point2f &point, const vector<Point2f>& points, double kernel_bandwidth)
{
	Point2f shifted_point = {0.0,0.0};

	double total_weight = 0;
	for (int i = 0; i < points.size();++i)
	{
		Point2f temp_point = points[i];
		double distance = Utils::euclidean_distance(point, temp_point);
		double weight = kernel_func(distance, kernel_bandwidth);
		shifted_point.x += temp_point.x*weight;
		shifted_point.y += temp_point.y*weight;
		
		total_weight += weight;
	}

	shifted_point.x /= total_weight;
	shifted_point.y /= total_weight;

	return shifted_point;
}

double Utils::euclidean_distance(const Point2f & point_a, Point2f & point_b)
{
	double total = 0;
	total = (point_a.x- point_b.x) * (point_a.x - point_b.x)
		+ (point_a.y - point_b.y) * (point_a.y - point_b.y);
	return sqrt(total);
}

double Utils::gaussian_kernel(double distance, double kernel_bandwidth)
{
	double temp = exp(-(distance*distance) / (kernel_bandwidth));
	return temp;
}

void Utils::getPoints(Mat& src, vector<Point2f>& points)
{
	points.clear();
	if (src.empty())
	{
		return;
	}
	for (int i = 0; i < src.rows;++i)
	{
		for (int j = 0; j < src.cols;++j)
		{
			if (src.at<ushort>(i,j)!=0)
			{
				points.push_back(Point2f(j, i));
			}
		}
	}
}

void Utils::displayCluster(Mat& displayImg, const vector<Point2f>& points, const vector<int> labels, const int labelscnt)
{
	assert(points.size() == labels.size());
	if (displayImg.empty())
	{
		displayImg.create(424, 512, CV_8UC3);
	}
	displayImg.setTo(Scalar::all(0));

	size_t pointCnt = points.size();
	vector<Scalar> colors(labelscnt+1);
	
	RNG rng(0xFFFFFFFF);
	for (size_t i = 1; i < labelscnt+1;++i)
	{
		colors[i] = Utils::randomColor(rng);
		rng.next();
	}
	Mat mask(displayImg.size(), CV_8U,Scalar::all(0));

	for (size_t i = 0; i < pointCnt;++i)
	{
		mask.at<uchar>(points[i]) = labels[i];
	}

	for (size_t i = 1; i < labelscnt+1;++i)
	{
		displayImg.setTo(colors[i], mask == i);
	}
	mask.release();
	colors.clear();
	
}

cv::Scalar Utils::randomColor(RNG rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void Utils::writeTocsv(const string& filename, const vector<Point2f>& points)
{
	ofstream outfile(filename,ios::binary);
	for (size_t i = 0; i < points.size();++i)
	{
		outfile << points[i].x << ',' << points[i].y << "\n";
	}
	outfile.close();
}
