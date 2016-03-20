#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include "opencvHeader.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/mean_shift/mean_shift.hpp"
using namespace std;
using namespace mlpack;
using namespace mlpack::meanshift;
namespace Utils
{
	class Histogram
	{
	public:
		static void getHist(cv::Mat& src, cv::Mat& depth_hist, const int histSize = 8000, const int dims = 1, const int channels = 0);
		static void plotHist(cv::Mat& hist, cv::Mat& showimg);
		static void getMaxPeaks(cv::Mat& hist, vector<pair<int, int> >&peaks);
		static void markMaxPeaks(cv::Mat& hist, cv::Mat& histimg);
	};

	class meanShift
	{
	public:
		meanShift() {};
		meanShift(double Radius,double maxIterations=1000):meanshift(Radius,maxIterations){}
		void Cluster(const vector<cv::Point2f>& points, vector<int>& labels, vector<cv::Point2f>& centroids);
	private:
		MeanShift<> meanshift;
	};
	double euclidean_distance(const cv::Point2f &point_a, cv::Point2f &point_b);
	void   getPoints(cv::Mat& src, vector<cv::Point2f>& points);
	void   displayCluster(cv::Mat& displayImg, const vector<cv::Point2f>& points, const vector<int> labels,const int labelscnt);
	inline cv::Scalar randomColor(cv::RNG rng);
	void   writeTocsv(const string& filename, const vector<cv::Point2f>& points);
	bool  findallfiles(const string& folderpath, vector<string>& files, string filetype);
	//void   meanshiftCluster(const vector<cv::Point2f>& points, vector<int>& labels, vector<cv::Point2f>& centroids);

	void NonMaximalSuppression(vector<cv::Rect>& rects, vector<double>& weights, float overlap,int etype=1);

	struct Detection {
		float score;
		cv::Rect bounds;
	};

	class NonMaximumSuppression
	{
	public:

		/** Type of the maximum computation per cluster. */
		static enum MaximumType {
			MAX_SCORE, ///< Take the detection with the highest score per cluster.
			AVERAGE, ///< Compute the average of the cluster.
			WEIGHTED_AVERAGE ///< Compute the average of the cluster, weighted by the score.
		};
		static float overlapThreshold;
		static int maximumType;
		static void eliminateRedundantDetections(vector<cv::Rect>& rects, vector<double>& weights, float overlap,int type=MaximumType::MAX_SCORE);
		static void sortByscores(vector<Detection>& candidates);
		static void cluster(vector<Detection>& candidates,vector<vector<Detection> >&clusters);
		static vector<Utils::Detection> extractOverlappingDetections(Detection& detection, vector<Detection>& candidates);
		static double computeOverlap(cv::Rect a, cv::Rect b);
		static void getMaxima(const vector<vector<Detection> >& clusters, vector<Detection>& finalDetections);
		static Detection getMaximum(const vector<Detection>& cluster);
	};
};
#endif
