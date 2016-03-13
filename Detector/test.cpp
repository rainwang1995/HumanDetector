#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "Utils.h"
#include "HONVNW.h"
#include "opencvHeader.h"
#include "preprocess.h"
#include <direct.h>
#include "preprocess.h"
#include <omp.h>
#include "Algorithm.hpp"
#include "SLTP.h"
#include <algorithm>
using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	//string path = "F:\\liuhao\\testTrainSet\\testset\\pos\\";
	string path = "F:\\liuhao\\testTrainSet\\posfull\\";
	//string path = "F:\\liuhao\\SZU-Kinect-People-Dataset\\depth\\";
	vector<string> files;
	Utils::findallfiles(path, files, "png");
	sort(files.begin(), files.end());
	DetectionAlgorithm* detector = new SLTP();
	//HONVNW honv;
	detector->loadSvmDetector("F:\\liuhao\\testTrainSet\\model\\sltpsvm2.xml");

	//Ptr<SVM> mysvm = StatModel::load<ml::SVM>("svm.xml");
	cout << "样本数 " << files.size() << endl;
	int cnt = 0;
	for (int i = 0; i < files.size();++i)
	{
		cout << files[i] << endl;
		string fullpath = path + files[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);

		//resize(sample, sample, Size(sample.cols / 3, sample.rows / 3),0.0,0.0, INTER_NEAREST);

		/*if (sample.rows != 128 || sample.cols != 64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}*/
		//resize(sample, sample, Size(128, 256), 0.0, 0.0, INTER_NEAREST);
		Mat filterimg;
		preProcessing::pixelFilter(sample, filterimg);
		//filterimg = sample;
		//vector<float> feature;
		//honv.compute(filterimg,feature);

		//Mat bing=Mat::zeros(480, 640, CV_16UC1);
		//filterimg.copyTo(bing(Rect(50, 50, filterimg.cols, filterimg.rows)));
		Mat imrgb;
		//Mat dd;
		//normalize(sample, dd);
		filterimg.convertTo(imrgb, CV_8U, 255.0/8000);
		cvtColor(imrgb,imrgb,CV_GRAY2BGR);

		//float response = mysvm->predict(feature);
		//cout << response << endl;
		vector<Rect> founds;
		vector<double> weights;
		//vector<Point> tp;
		detector->detectMultiScale(filterimg, founds, weights);
		
		//detector->detect(filterimg, tp, weights);
		//honv.detect(bing, tp, weights);
		//cout << "in main:" << endl;
		//cout << founds.size()<<endl;
		
		for (int j = 0; j < founds.size();++j)
		{
			++cnt;
			Rect r = founds[j];
			//cout << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
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
		cvWaitKey(33);
		cout << endl;
	}
	cout << "检测到人体数 " << cnt<<endl;
	cout << "true positive: " << (float)cnt / files.size() << endl;
	std::system("pause");
}
