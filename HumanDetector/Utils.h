#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include "opencvHeader.h"
using namespace std;
namespace Utils
{
	class Histogram
	{
	public:
		static void getHist(Mat& src, Mat& depth_hist, const int histSize = 8000, const int dims = 1, const int channels = 0);
		static void plotHist(Mat& hist, Mat& showimg);
		static void getMaxPeaks(Mat& hist, vector<pair<int, int> >&peaks);
		static void markMaxPeaks(Mat& hist, Mat& histimg);
	};

	class MeanShift
	{
	public:
		MeanShift() { set_kernel(NULL); }
		MeanShift(double(*_kernel_func)(double, double)) { set_kernel(_kernel_func); }
		void cluster(const vector<Point2f>& points, vector<Point2f>& centers,vector<int>& labels,double kernel_bandwidth=2);
	private:
		double(*kernel_func)(double, double);
		void set_kernel(double(*_kernel_func)(double, double));
		Point2f shift_point(const Point2f &, const vector<Point2f > &, double);
	};

	double euclidean_distance(const Point2f &point_a, Point2f &point_b);
	double gaussian_kernel(double distance, double kernel_bandwidth);
	void   getPoints(Mat& src, vector<Point2f>& points);
	void   displayCluster(Mat& displayImg, const vector<Point2f>& points, const vector<int> labels,const int labelscnt);
	inline Scalar randomColor(RNG rng);
	void   writeTocsv(const string& filename, const vector<Point2f>& points);
};
#endif
