#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "Utils.h"
#include "HONVNW.h"
#include "opencvHeader.h"
#include "preprocess.h"
#include <direct.h>
#include "preprocess.h"
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	string path = "F:\\liuhao\\SZU-Kinect-People-Dataset\\Pos\\P2\\";
	//string path = "F:\\liuhao\\SZU-Kinect-People-Dataset\\depthclip\\P1\\";
	vector<string> files;
	Utils::findallfiles(path, files, "png");

	HONVNW honv;
	honv.setSvmDetector("svm.xml");

	Ptr<SVM> mysvm = StatModel::load<ml::SVM>("svm.xml");

	for (int i = 0; i < files.size();++i)
	{
		string fullpath = path + files[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);

		//resize(sample, sample, Size(sample.cols / 3, sample.rows / 3),0.0,0.0, INTER_NEAREST);

		/*if (sample.rows != 128 || sample.cols != 64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}*/
		Mat filterimg;
		preProcessing::pixelFilter(sample, filterimg);
		//vector<float> feature;
		//honv.compute(filterimg,feature);

		//Mat bing=Mat::zeros(480, 640, CV_16UC1);
		//filterimg.copyTo(bing(Rect(50, 50, filterimg.cols, filterimg.rows)));
		Mat imrgb;
		filterimg.convertTo(imrgb, CV_8U, 255.0 / 4096);
		cvtColor(imrgb,imrgb,CV_GRAY2BGR);

		//float response = mysvm->predict(feature);
		//cout << response << endl;
		vector<Rect> founds;
		vector<double> weights;
		//vector<Point> tp;
		honv.detectMultiScale(filterimg, founds, weights);
		
		//honv.detect(bing, tp, weights);
		cout << "in main:" << endl;
		cout << founds.size()<<endl;
		//Mat imrgb;
		//img.convertTo(imrgb, CV_8U, 255.0 / 4096);
		//cvtColor(imrgb,imrgb,CV_GRAY2BGR);
		//cout << founds.size() << endl;
		
		for (int j = 0; j < founds.size();++j)
		{
			Rect r = founds[j];
			cout << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > filterimg.cols)
				r.width = filterimg.cols - r.x;
			if (r.y + r.height > filterimg.rows)
				r.height = filterimg.rows - r.y;

			rectangle(imrgb, r, Scalar(0, 0, 255),4);
		}

		cv::imshow("test", imrgb);
		cvWaitKey();
		cout << endl;
	}
	
	system("pause");
}
