#include <iostream>
#include "opencvHeader.h"
#include "CKinect.h"
#include "preprocess.h"
#include "Utils.h"
#include <fstream>
#include <string>
#include "HONVNW.h"
#include "Graphpropagation.h"
using namespace std;
using namespace cv;

int main()
{
	//test gp
	//
	vector<Rect> foundsx;
	foundsx.push_back(Rect(5,5,10,10));
	foundsx.push_back(Rect(6, 6, 11, 11));
	foundsx.push_back(Rect(5, 5, 12, 12));
	foundsx.push_back(Rect(7, 7, 9, 9));

	vector<double> weightsx;
	weightsx.push_back(0.5);
	weightsx.push_back(0.4);
	weightsx.push_back(0.8);
	weightsx.push_back(0.4);
	GraphProgation gp;

	vector<Rect> frec;
	double t = (double)getTickCount();
	gp.mergeRects(foundsx, weightsx, frec);
	cout << ((double)getTickCount()-t) / getTickFrequency() << endl;
	//


	CKinect kinetCtrl;
	if (!kinetCtrl.Init())
	{
		cerr << "Kinect Init failed" << endl;
		return 1;
	}
	int width = kinetCtrl.getDepthWidth();
	int height = kinetCtrl.getDepthHeight();
	Mat depthimg(height, width, CV_16UC1);
	ushort* depthdata = (ushort*)depthimg.data;
	/*while (true)
	{
		kinetCtrl.UpdateDepth(depthdata);
		Mat depth8U(height, width, CV_8U);
		depthimg.convertTo(depth8U, CV_8U, 255 / 8000.0);
		imshow("depth", depth8U);
		
		Mat filterd;
		preProcessing::pixelFilter(depthimg, filterd);
		filterd.convertTo(depth8U, CV_8U, 255 / 8000.0);
		imshow("depthfilterd", depth8U);
		Mat hist;
		Utils::Histogram::getHist(filterd, hist, 80);
		Mat histimg;
		Utils::Histogram::markMaxPeaks(hist, histimg);
		imshow("hist", histimg);
		vector<pair<int, int> > peaks;
		Utils::Histogram::getMaxPeaks(hist, peaks);
		int index = 0;
		int maxbinval = 0;
		for (int i = 0; i < peaks.size();++i)
		{
			if (peaks[i].first>maxbinval)
			{
				maxbinval = peaks[i].first;
				index = peaks[i].second;
			}
		}
		cout << index <<" "<<maxbinval<< endl;
		Mat seg = filterd.clone();
		seg.setTo(0, seg < 100 * index);
		seg.setTo(0, seg > 100 * (index + 1));
		seg.convertTo(depth8U, CV_8U, 255 / 8000.0);
		imshow("seg", depth8U);

		int key=waitKey(10);
		if (key=='q')
		{
			break;
		}
		if (key=='s')
		{
			imwrite("hist.png", histimg);
		}
	}*/
	depthimg = imread("depth000495.png", IMREAD_ANYDEPTH);
	Mat depth8U(height, width, CV_8U);
	depthimg.convertTo(depth8U, CV_8U, 255 / 8000.0);
	imshow("depth", depth8U);

	Mat filterd;
	preProcessing::pixelFilter(depthimg, filterd);
	filterd.convertTo(depth8U, CV_8U, 255 / 8000.0);
	imshow("depthfilterd", depth8U);
	Mat hist;
	Utils::Histogram::getHist(filterd, hist, 80);
	Mat histimg;
	Utils::Histogram::markMaxPeaks(hist, histimg);
	imshow("hist", histimg);
	vector<pair<int, int> > peaks;
	Utils::Histogram::getMaxPeaks(hist, peaks);

	imwrite("hist.png", histimg);

	int index = 0;
	int maxbinval = 0;
	Mat clusterImg(depthimg.size(), CV_8UC3);
	Utils::meanShift meanshift;
	for (int i = 0; i < peaks.size(); ++i)
	{
		index = peaks[i].second;
		Mat seg = filterd.clone();
		seg.setTo(0, seg < 100 * index);
		seg.setTo(0, seg > 100 * (index + 1));
		seg.convertTo(depth8U, CV_8U, 255 / 8000.0);

		vector<Point2f> points,centers;
		vector<int> labels;
		Utils::getPoints(seg, points);
		cout << points.size() << endl;
		meanshift.Cluster(points, labels, centers);
		//cout << centers.size() << endl;
		clusterImg.setTo(Scalar::all(0));
		Utils::displayCluster(clusterImg, points, labels, centers.size());
		//meanshift.Cluster(points, labels, centers);
		imshow("seg", depth8U);
		imshow("meanshift", clusterImg);
		int key = waitKey();
		if (key=='s')
		{
			Utils::writeTocsv("data.csv", points);
		}
		
	}
	
	return 0;
}