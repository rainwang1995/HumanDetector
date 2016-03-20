#include "Utils.h"
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <io.h>

#define EPSILON 0.0000001
using namespace cv;

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

double Utils::euclidean_distance(const Point2f & point_a, Point2f & point_b)
{
	double total = 0;
	total = (point_a.x- point_b.x) * (point_a.x - point_b.x)
		+ (point_a.y - point_b.y) * (point_a.y - point_b.y);
	return sqrt(total);
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
		mask.at<uchar>(points[i]) = labels[i]+1;
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

bool Utils::findallfiles(const string& folderpath, vector<string>& files,string filetype)
{
		_finddata_t fileInfo;
		string strfind = folderpath + "\\*."+filetype;
		//replacepath(strfind);
		intptr_t handle = _findfirst(strfind.c_str(), &fileInfo);

		if (handle == -1)
		{
			cerr << "failded to open folder" << folderpath << endl;
			return false;
		}
		
		files.push_back(fileInfo.name);

		while (_findnext(handle, &fileInfo) == 0)
		{
			if (fileInfo.attrib & _A_SUBDIR)
				continue;

			files.push_back(fileInfo.name);

		}
		_findclose(handle);
		return true;
}

void Utils::NonMaximalSuppression(vector<cv::Rect>& rects, vector<double>& weights, float overlap,int etype)
{
	
	Utils::NonMaximumSuppression::eliminateRedundantDetections(rects, weights, overlap, etype);

	
}

void Utils::meanShift::Cluster(const vector<Point2f>& points, vector<int>& labels, vector<Point2f>& centers)
{
	arma::mat dataset(2,points.size());
	for (int i = 0; i < points.size();++i)
	{
		dataset(0, i) = points[i].x;
		dataset(1, i) = points[i].y;
	}

	arma::Col<size_t> assignments;
	arma::mat cenroids;
	meanshift.Cluster(dataset, assignments, cenroids);
	labels=arma::conv_to<vector<int> >::from(assignments);
	centers.resize(cenroids.n_cols);
	centers.reserve(centers.size());
	for (int i = 0; i < cenroids.n_cols;++i)
	{
		centers[i]=Point2f(cenroids(0, i), cenroids(1, i));
	}
}

float Utils::NonMaximumSuppression::overlapThreshold = 1.0;
int Utils::NonMaximumSuppression::maximumType = 0;
void Utils::NonMaximumSuppression::eliminateRedundantDetections(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int type)
{
	if (overlap>=1)
	{
		return;
	}
	overlapThreshold = overlap;
	maximumType = type;
	vector<Detection> candidates(rects.size());
	for (int i = 0; i < rects.size();++i)
	{
		candidates[i].bounds = rects[i];
		candidates[i].score = weights[i];
	}
	sortByscores(candidates);
	vector<vector<Detection> > clusters;
	cluster(candidates, clusters);
	getMaxima(clusters, candidates);

	rects.clear();
	weights.clear();
	for (int i = 0; i < candidates.size();++i)
	{
		rects.push_back(candidates[i].bounds);
		weights.push_back(candidates[i].score);
	}

}

void Utils::NonMaximumSuppression::sortByscores(vector<Detection>& candidates)
{
	std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
		return a.score < b.score;
	});
}

void Utils::NonMaximumSuppression::cluster(vector<Detection>& candidates, vector<vector<Detection> >&clusters)
{
	clusters.clear();
	while (!candidates.empty())
		clusters.push_back(extractOverlappingDetections(candidates.back(), candidates));
}

vector<Utils::Detection> Utils::NonMaximumSuppression::extractOverlappingDetections(Detection& detection, vector<Detection>& candidates)
{
	vector<Detection> overlappingDetections;
	auto firstOverlapping = std::stable_partition(candidates.begin(), candidates.end(), [&](const Detection& candidate) {
		return computeOverlap(detection.bounds, candidate.bounds) <= overlapThreshold;
	});
	std::move(firstOverlapping, candidates.end(), std::back_inserter(overlappingDetections));
	std::reverse(overlappingDetections.begin(), overlappingDetections.end());
	candidates.erase(firstOverlapping, candidates.end());
	return overlappingDetections;
}

double Utils::NonMaximumSuppression::computeOverlap(cv::Rect a, cv::Rect b)
{
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

void Utils::NonMaximumSuppression::getMaxima(const vector<vector<Detection> >& clusters, vector<Detection>& finalDetections)
{
	finalDetections.clear();
	finalDetections.reserve(clusters.size());
	for (const vector<Detection>& cluster : clusters)
		finalDetections.push_back(getMaximum(cluster));
}

Utils::Detection Utils::NonMaximumSuppression::getMaximum(const vector<Detection>& cluster)
{
	if (maximumType == MaximumType::MAX_SCORE) {
		return cluster.front();
	}
	else if (maximumType == MaximumType::AVERAGE) {
		double xSum = 0;
		double ySum = 0;
		double wSum = 0;
		double hSum = 0;
		for (const Detection& elem : cluster) {
			xSum += elem.bounds.x;
			ySum += elem.bounds.y;
			wSum += elem.bounds.width;
			hSum += elem.bounds.height;
		}
		int x = static_cast<int>(std::round(xSum / cluster.size()));
		int y = static_cast<int>(std::round(ySum / cluster.size()));
		int w = static_cast<int>(std::round(wSum / cluster.size()));
		int h = static_cast<int>(std::round(hSum / cluster.size()));
		float score = cluster.front().score;
		Rect averageBounds(x, y, w, h);
		return Detection{ score, averageBounds };
	}
	else if (maximumType == MaximumType::WEIGHTED_AVERAGE) {
		double weightSum = 0;
		double xSum = 0;
		double ySum = 0;
		double wSum = 0;
		double hSum = 0;
		for (const Detection& elem : cluster) {
			double weight = elem.score;
			weightSum += weight;
			xSum += weight * elem.bounds.x;
			ySum += weight * elem.bounds.y;
			wSum += weight * elem.bounds.width;
			hSum += weight * elem.bounds.height;
		}
		int x = static_cast<int>(std::round(xSum / weightSum));
		int y = static_cast<int>(std::round(ySum / weightSum));
		int w = static_cast<int>(std::round(wSum / weightSum));
		int h = static_cast<int>(std::round(hSum / weightSum));
		float score = cluster.front().score;
		Rect averageBounds(x, y, w, h);
		return Detection{ score, averageBounds };
	}
	else {
		throw std::runtime_error("NonMaximumSuppression: unsupported maximum type");
	}

}
