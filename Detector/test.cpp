#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "Utils.h"
//#include "HONVNW.h"
#include "opencvHeader.h"
#include "preprocess.h"
#include <direct.h>
#include "preprocess.h"
#include <omp.h>
#include "Algorithm.hpp"
#include "SLTP.h"
#include <algorithm>
#include "LTDP.h"
#include "HDD.h"
#include "HONVNW.h"
#include <fstream>
#include <sstream>
#include "Graphpropagation.h"
#include "SLTP2.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

void testgp()
{
	vector<string> detectorpath(4);
	detectorpath[3] = "F:\\liuhao\\testTrainSet\\model\\sltpsvmnofill1.xml";
	detectorpath[1] = "F:\\liuhao\\testTrainSet\\model\\hddsvmnofill1.xml";
	detectorpath[2] = "F:\\liuhao\\testTrainSet\\model\\honvsvmnofill1.xml";
	detectorpath[0] = "F:\\liuhao\\testTrainSet\\model\\ltdpsvmnofill1.xml";

	GraphProgation *gp=new GraphProgation();
	gp->setSvmDetectors(detectorpath);

	string path = "F:\\liuhao\\testTrainSet\\hardtest\\";
	vector<string> files;
	Utils::findallfiles(path, files, "png");
	sort(files.begin(), files.end());

	cout << "样本数 " << files.size() << endl;
	double sumt = 0;
	for (int i = 0; i < files.size(); ++i)
	{
		cout << files[i] << endl;
		string fullpath = path + files[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
		if (sample.empty())
			continue;

		Mat filterimg;
		preProcessing::pixelFilter(sample, filterimg);
		Mat imrgb;
		
		filterimg.convertTo(imrgb, CV_8U, 255.0 / 8000);
		cvtColor(imrgb, imrgb, CV_GRAY2BGR);

		vector<Rect> founds;
		vector<double> weights;
		double t = (double)getTickCount();
		gp->detect(filterimg, founds, weights);
		t = ((double)getTickCount() - t) / getTickFrequency();
		sumt += t;
		//Utils::NonMaximalSuppression(founds, weights, 0.5);

		for (int j = 0; j < founds.size(); ++j)
		{
			Rect r = founds[j];

			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > filterimg.cols)
				r.width = filterimg.cols - r.x;
			if (r.y + r.height > filterimg.rows)
				r.height = filterimg.rows - r.y;

			rectangle(imrgb, r, Scalar(0, 0, 255), 4);
		}

		cv::imshow("test", imrgb);
		cvWaitKey(10);
		//cout << endl;
	}

	delete gp;
}

void testfromKinect()
{
	DetectionAlgorithm* detector = new LTDP();
	//detector = new HDD();
	detector = new SLTP();
	//detector = new HONVNW();
	detector->loadSvmDetector("F:\\liuhao\\testTrainSet\\model\\sltpsvmf.xml");
	//detector->set_signThreshold(100);
	//detector->set_signThreshold(50);
	VideoCapture capture(CV_CAP_OPENNI);
	capture.set(CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION, 0);
	Mat depthmap, filterimg;

	while (false)
	{
		capture.grab();
		capture.retrieve(depthmap, CV_CAP_OPENNI_DEPTH_MAP);
		if (depthmap.empty())
		{
			continue;
		}
		Mat inpaintmap;
		preProcessing::pixelFilter(depthmap, filterimg);

		Mat imrgb;
		filterimg.convertTo(imrgb, CV_8U, 255.0 / 10000);
		cvtColor(imrgb, imrgb, CV_GRAY2BGR);

		vector<Rect> founds;
		vector<double> weights;
		detector->detectMultiScale(filterimg, founds, weights, 0.8);

		for (int j = 0; j < founds.size(); ++j)
		{

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

			rectangle(imrgb, r, Scalar(0, 0, 255), 4);
		}

		cv::imshow("test", imrgb);
		//cvWaitKey(33);
		char key = waitKey(10);
		//imshow("depth", depthmap);
		if (key == 'q')
		{
			break;
		}
		else if (key == 's')
		{
			imwrite("1.png", depthmap);
		}
	}
}

void testfromFiles()
{
	DetectionAlgorithm* detector = new LTDP();
	
	detector = new HDD();
	//DetectionAlgorithm *detector = new SLTP();
	//detector = new HONVNW();
	//detector = new SLTP();
	detector->loadSvmDetector("F:\\liuhao\\testTrainSet\\model\\hddsvmnofill1.xml");
	//detector->set_signThreshold(100);
	//detector->set_signThreshold(50);
	string path = "F:\\liuhao\\testTrainSet\\negtest\\";
	vector<string> files;
	Utils::findallfiles(path, files, "png");
	sort(files.begin(), files.end());

	cout << "样本数 " << files.size() << endl;
	int cnt = 0;
	double sumt = 0.0;
	//输出结果
	//fstream fout("F:\\liuhao\\testTrainSet\\reslut\\honvneg1.txt", ios::app);
	//fstream fouttime("F:\\liuhao\\testTrainSet\\reslut\\time.txt", ios::app);
	for (int i = 0; i < files.size(); ++i)
	{
		cout << files[i] << endl;
		string fullpath = path + files[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
		if (sample.empty())
			continue;

		if (sample.rows != 128 || sample.cols != 64)
		{
		resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}
		//resize(sample, sample, Size(128, 256), 0.0, 0.0, INTER_NEAREST);
		Mat filterimg;
		preProcessing::pixelFilter(sample, filterimg);
		//filterimg = sample;
		//vector<float> feature;
		Mat painted;
		//preProcessing::inpaintrawdepth(filterimg, painted, 3, 0.5);
		//Mat bing=Mat::zeros(480, 640, CV_16UC1);
		//filterimg.copyTo(bing(Rect(50, 50, filterimg.cols, filterimg.rows)));
		Mat imrgb;
		//Mat dd;
		//normalize(sample, dd);
		filterimg.convertTo(imrgb, CV_8U, 255.0 / 8000);
		cvtColor(imrgb, imrgb, CV_GRAY2BGR);

		vector<Rect> founds;
		vector<double> weights;
		double t = (double)getTickCount();
		detector->detectMultiScale(filterimg, founds, weights, 0.);
		t = ((double)getTickCount() - t) / getTickFrequency();
		sumt += t;
		//Utils::NonMaximalSuppression(founds, weights, 0.5);

		for (int j = 0; j < founds.size(); ++j)
		{
			//fout << files[i] << " " << weights[j] << " "
			//<< founds[j].x << " " << founds[j].y
			//<< " " << founds[j].width << " " << founds[j].height << endl;
			++cnt;
			Rect r = founds[j];

			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > filterimg.cols)
				r.width = filterimg.cols - r.x;
			if (r.y + r.height > filterimg.rows)
				r.height = filterimg.rows - r.y;

			rectangle(imrgb, r, Scalar(0, 0, 255), 4);
		}

		//cv::imshow("test", imrgb);
		//cvWaitKey(10);
		//cout << endl;
	}

	//fouttime << "hdd2: " << sumt / files.size() << endl;
	//fout.close();
	//fouttime.close();

	cout << "time per image" << sumt / files.size() << endl;
	cout << "检测到人体数 " << cnt << endl;
}

void testfrommultiDetectors(DetectionAlgorithm* detectors[], string detectornames[], const string modelpath,const string& testpath,const string& testnegpath, const string& resultpath, void(*pfunc)(Mat&, Mat&), string testtype)
{
	_mkdir(resultpath.c_str());
	vector<string> testfiles;
	Utils::findallfiles(testpath, testfiles,"png");
	sort(testfiles.begin(), testfiles.end());
	cout << "正样本样本数 " << testfiles.size() << endl;

	vector<string> negfiles;
	Utils::findallfiles(testnegpath, negfiles, "png");
	cout << "负样本数 " << negfiles.size() << endl;
	cout << testtype << endl;
//#pragma omp parallel for
	for (int j = 4; j < 5;++j)
	{
		cout << detectornames[j] << endl;
		for (int k = 0; k < 2;++k)
		{
			string svmpath = modelpath + detectornames[j] + "_" + testtype;
			svmpath += (k == 0 ? "svm1.xml" : "svm2.xml");
			detectors[j]->loadSvmDetector(svmpath);

			string outputpath = resultpath + detectornames[j] + "_" + testtype;
			outputpath += (k == 0 ? "3.txt" : "4.txt");

			//输出结果
			fstream fout(outputpath, ios::app);
			for (int i = 0; i < testfiles.size(); ++i)
			{
				//cout << testpath[i] << endl;
				string fullpath = testpath + testfiles[i];
				Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
				if (sample.empty())
					continue;
				//if (sample.rows != 128 || sample.cols != 64)
				//{
				//	resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
				//}
				//resize(sample, sample, Size(128, 256), 0.0, 0.0, INTER_NEAREST);
				Mat filterimg;
				if (pfunc == NULL)
				{
					filterimg = sample;
				}
				else
					pfunc(sample, filterimg);

				vector<Rect> founds;
				vector<double> weights;
				detectors[j]->detectMultiScale(filterimg, founds, weights, 0.);

				Mat imrgb;
				filterimg.convertTo(imrgb, CV_8U, 255.0 / 8000);
				cvtColor(imrgb, imrgb, CV_GRAY2BGR);

				for (int h = 0; h < founds.size(); ++h)
				{
					Rect r = founds[h];

					if (r.x < 0)
						r.x = 0;
					if (r.y < 0)
						r.y = 0;
					if (r.x + r.width > filterimg.cols)
						r.width = filterimg.cols - r.x;
					if (r.y + r.height > filterimg.rows)
						r.height = filterimg.rows - r.y;

					fout << testfiles[i] << " " << weights[h] << " "
					<< r.x << " " << r.y
					<< " " << r.width << " " << r.height << endl;

					rectangle(imrgb, r, Scalar(0, 0, 255), 4);

				}

				cv::imshow("test", imrgb);
				cvWaitKey(10);
			}

			//负样本测试
//#pragma omp parallel for
			for (int i = 0; i < negfiles.size(); ++i)
			{
				//cout << testpath[i] << endl;
				string fullpath = testnegpath + negfiles[i];
				Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
				if (sample.empty())
					continue;
				//if (sample.rows != 128 || sample.cols != 64)
				//{
				//	resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
				//}
				//resize(sample, sample, Size(128, 256), 0.0, 0.0, INTER_NEAREST);
				Mat filterimg;
				if (pfunc == NULL)
				{
					filterimg = sample;
				}
				else
					pfunc(sample, filterimg);

				vector<Rect> founds;
				vector<double> weights;
				detectors[j]->detectMultiScale(filterimg, founds, weights, 0.);

				for (int h = 0; h < founds.size(); ++h)
				{
					Rect r = founds[h];

					if (r.x < 0)
						r.x = 0;
					if (r.y < 0)
						r.y = 0;
					if (r.x + r.width > filterimg.cols)
						r.width = filterimg.cols - r.x;
					if (r.y + r.height > filterimg.rows)
						r.height = filterimg.rows - r.y;

					fout << negfiles[i] << " " << weights[h] << " "
						<< r.x << " " << r.y
						<< " " << r.width << " " << r.height << endl;
				}
			}

			fout.close();
		}		
	}
}

int main()
{
	//testfromFiles();
	//waitKey();
	//testgp();

	omp_set_num_threads(2);
	string testpath = "F:\\liuhao\\testTrainSet\\hardtest\\";
	string resultspath = "F:\\liuhao\\testTrainSet\\results2\\";
	string modelpath = "F:\\liuhao\\testTrainSet\\models2\\";
	string negpath = "F:\\liuhao\\testTrainSet\\negtestfull\\";
	DetectionAlgorithm* detectors[5];
	detectors[0] = new SLTP();
	detectors[1] = new HDD();
	detectors[2] = new LTDP();
	detectors[3] = new HONVNW();
	detectors[4] = new SLTP2();
	string detectornames[5] = { "SLTP","HDD","LTDP","HONV","SLTP2" };
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, NULL, "ori");
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc1, "NN");
	testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc2, "MH");
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc3, "INPAINT");
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc4, "NNMH");

	for (int i = 0; i < 4;++i)
	{
		delete detectors[i];
	}

	std::system("pause");
}
