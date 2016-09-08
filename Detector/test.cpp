#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "Utils.h"
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
#include "RDSF.h"
#include "LDPK.h"
#include "PLDPK.h"
#include "ccafusion.h"
#include "PELDP.h"
#include "ELDP.h"
#include "MPLDPK.h"
//#include "CKinect.h"
#include <windows.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

void testgp(DetectionAlgorithm* detectors[], string detectornames[], const string modelpath,
	const string& testpath, const string& testnegpath,
	const string& resultpath, string testtypes[], int K[])
{
	_mkdir(resultpath.c_str());
	vector<string> testfiles;
	Utils::findallfiles(testpath, testfiles, "png");
	sort(testfiles.begin(), testfiles.end());
	cout << "正样本样本数 " << testfiles.size() << endl;

	vector<string> negfiles;
	Utils::findallfiles(testnegpath, negfiles, "png");
	cout << "负样本数 " << negfiles.size() << endl;

	for (int j = 0; j < 3; ++j)
	{
		//for (int k = 0; k < 1;++k)
		{
			string svmpath = modelpath + detectornames[j] + "_" + testtypes[j];
			svmpath += (K[j] == 0 ? "svm1.xml" : "svm2.xml");
			detectors[j]->loadSvmDetector(svmpath);
		}
	}
	/* string matrixpath = modelpath + "ccamatrix.yml";
	detectors[2]->loadccamatrix(matrixpath);*/

	GraphProgation *gp = new GraphProgation(3);
	gp->setSvmDetectors(detectors);

	string output = resultpath + "gp_ori" + ".txt";
	ofstream fout(output);
	for (int i = 0; i < testfiles.size(); ++i)
	{
		if (testfiles[i] != "img_0258.png")
		{
			continue;
		}
		cout << testfiles[i] << endl;
		string fullpath = testpath + testfiles[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
		if (sample.empty())
			continue;

		vector<Rect> founds;
		vector<double> weights;
		gp->detect(sample, founds, weights, testtypes);
		Mat imrgb;
		sample.convertTo(imrgb, CV_8U, 255.0 / 8000);
		cvtColor(imrgb, imrgb, CV_GRAY2BGR);

		if (founds.size() == 0 || weights[0]<0.5)
		{
			cout << testfiles[i] << endl;
			cvWaitKey();
		}

		for (int k = 0; k < founds.size(); ++k)
		{
			cout << weights[k] << endl;
			/*if (weights[k]<0.5)
			{

			continue;
			}*/
			Rect r = founds[k];

			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > sample.cols)
				r.width = sample.cols - r.x;
			if (r.y + r.height > sample.rows)
				r.height = sample.rows - r.y;

			fout << testfiles[i] << " " << weights[k] << " "
				<< r.x << " " << r.y
				<< " " << r.width << " " << r.height << endl;

			rectangle(imrgb, r, Scalar(0, 0, 255), 4);
		}

		cv::imshow("test", imrgb);
		cvWaitKey();
	}

	//for (int i = 0; i < negfiles.size(); ++i)
	//{
	//	cout << negfiles[i] << endl;

	//	string fullpath = testnegpath + negfiles[i];
	//	Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
	//	if (sample.empty())
	//		continue;

	//	vector<Rect> founds;
	//	vector<double> weights;
	//	gp->detect(sample, founds, weights);

	//	for (int j = 0; j < founds.size(); ++j)
	//	{
	//		Rect r = founds[j];

	//		if (r.x < 0)
	//			r.x = 0;
	//		if (r.y < 0)
	//			r.y = 0;
	//		if (r.x + r.width > sample.cols)
	//			r.width = sample.cols - r.x;
	//		if (r.y + r.height > sample.rows)
	//			r.height = sample.rows - r.y;

	//		fout << negfiles[i] << " " << weights[j] << " "
	//			<< r.x << " " << r.y
	//			<< " " << r.width << " " << r.height << endl;
	//		//rectangle(imrgb, r, Scalar(0, 0, 255), 4);
	//	}
	//}

	fout.close();
}


//void testgp()
//{
//	vector<string> detectorpath(4);
//	detectorpath[3] = "F:\\liuhao\\testTrainSet\\model\\sltpsvmnofill1.xml";
//	detectorpath[1] = "F:\\liuhao\\testTrainSet\\model\\hddsvmnofill1.xml";
//	detectorpath[2] = "F:\\liuhao\\testTrainSet\\model\\honvsvmnofill1.xml";
//	detectorpath[0] = "F:\\liuhao\\testTrainSet\\model\\ltdpsvmnofill1.xml";
//
//	GraphProgation *gp = new GraphProgation(4);
//	gp->setSvmDetectors(detectorpath);
//
//	string path = "F:\\liuhao\\testTrainSet\\hardtest\\";
//	vector<string> files;
//	Utils::findallfiles(path, files, "png");
//	sort(files.begin(), files.end());
//
//	cout << "样本数 " << files.size() << endl;
//	double sumt = 0;
//	for (int i = 0; i < files.size(); ++i)
//	{
//		cout << files[i] << endl;
//		string fullpath = path + files[i];
//		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
//		if (sample.empty())
//			continue;
//
//		Mat filterimg;
//		preProcessing::pixelFilter(sample, filterimg);
//		Mat imrgb;
//
//		filterimg.convertTo(imrgb, CV_8U, 255.0 / 8000);
//		cvtColor(imrgb, imrgb, CV_GRAY2BGR);
//
//		vector<Rect> founds;
//		vector<double> weights;
//		double t = (double)getTickCount();
//		gp->detect(filterimg, founds, weights);
//		t = ((double)getTickCount() - t) / getTickFrequency();
//		sumt += t;
//		//Utils::NonMaximalSuppression(founds, weights, 0.5);
//
//		for (int j = 0; j < founds.size(); ++j)
//		{
//			Rect r = founds[j];
//
//			if (r.x < 0)
//				r.x = 0;
//			if (r.y < 0)
//				r.y = 0;
//			if (r.x + r.width > filterimg.cols)
//				r.width = filterimg.cols - r.x;
//			if (r.y + r.height > filterimg.rows)
//				r.height = filterimg.rows - r.y;
//
//			rectangle(imrgb, r, Scalar(0, 0, 255), 4);
//		}
//
//		cv::imshow("test", imrgb);
//		cvWaitKey(10);
//		//cout << endl;
//	}
//
//	delete gp;
//}

//void testfromKinect()
//{
//	DetectionAlgorithm* detector = new HDD();
//
//	detector->loadSvmDetector("F:\\liuhao\\testTrainSet\\models2\\HDD_orisvm1.xml");
//	//detector->set_signThreshold(100);
//	//detector->set_signThreshold(50);
//	VideoCapture capture(CV_CAP_OPENNI);
//	capture.set(CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION, 0);
//	Mat depthmap, filterimg;
//
//	while (true)
//	{
//		capture.grab();
//		capture.retrieve(depthmap, CV_CAP_OPENNI_DEPTH_MAP);
//		if (depthmap.empty())
//		{
//			continue;
//		}
//		filterimg = depthmap;
//
//		Mat imrgb;
//		filterimg.convertTo(imrgb, CV_8U, 255.0 / 10000);
//		cvtColor(imrgb, imrgb, CV_GRAY2BGR);
//
//		vector<Rect> founds;
//		vector<double> weights;
//		detector->detectMultiScale(filterimg, founds, weights, 0.);
//
//		for (int j = 0; j < founds.size(); ++j)
//		{
//
//			Rect r = founds[j];
//			//cout << r.x << " " << r.y << " " << r.width << " " << r.height << endl;
//			if (r.x < 0)
//				r.x = 0;
//			if (r.y < 0)
//				r.y = 0;
//			if (r.x + r.width > filterimg.cols)
//				r.width = filterimg.cols - r.x;
//			if (r.y + r.height > filterimg.rows)
//				r.height = filterimg.rows - r.y;
//
//			rectangle(imrgb, r, Scalar(0, 0, 255), 4);
//		}
//
//		cv::imshow("test", imrgb);
//		//cvWaitKey(33);
//		char key = waitKey(10);
//		//imshow("depth", depthmap);
//		if (key == 'q')
//		{
//			break;
//		}
//		else if (key == 's')
//		{
//			imwrite("1.png", depthmap);
//		}
//	}
//}
//
//void testfromKinect2(DetectionAlgorithm* detector, string svmpath)
//{
//	CKinect kinetCtrl;
//	if (!kinetCtrl.Init())
//	{
//		cerr << "Kinect Init failed" << endl;
//		return;
//	}
//
//	detector->loadSvmDetector(svmpath);
//
//	int width = kinetCtrl.getDepthWidth();
//	int height = kinetCtrl.getDepthHeight();
//	Mat depthimg(height, width, CV_16UC1);
//	ushort* depthdata = (ushort*)depthimg.data;
//	while (true)
//	{
//		kinetCtrl.UpdateDepth(depthdata);
//		Mat depth8U(height, width, CV_8U);
//		depthimg.convertTo(depth8U, CV_8U, 255 / 8000.0);
//		cvtColor(depth8U, depth8U, CV_GRAY2BGR);
//		imshow("depth", depth8U);
//
//		Mat filterd;
//		filterfunc2(depthimg, filterd);
//		vector<Rect> founds;
//		vector<double> weights;
//
//		//double t1 = (double)getTickCount();
//		detector->detectMultiScale(filterd, founds, weights, 0.);
//
//		for (int h = 0; h < founds.size(); ++h)
//		{
//			Rect r = founds[h];
//
//			if (r.x < 0)
//				r.x = 0;
//			if (r.y < 0)
//				r.y = 0;
//			if (r.x + r.width > filterd.cols)
//				r.width = filterd.cols - r.x;
//			if (r.y + r.height > filterd.rows)
//				r.height = filterd.rows - r.y;
//
//			//if(weights[h]>=0.5)
//			rectangle(depth8U, r, Scalar(0, 0, 255), 4);
//		}
//		imshow("test", depth8U);
//		//Utils::NonMaximalSuppression2(founds, weights, 0.5, 0);
//		int key = waitKey(10);
//		if (key == 'q')
//		{
//			break;
//		}
//	}
//
//	kinetCtrl.Release();
//}

void readcca(const string& path, vector<string>& names)
{
	ifstream readfile(path);
	if (!readfile.is_open())
	{
		cerr << "error file" << endl;
	}
	string txtline;
	while (getline(readfile,txtline))
	{
		istringstream istr(txtline);
		string filename;
		double weight;
		int x, y, width, height;
		istr >> filename >> weight >> x >> y >> width >> height;
		names.push_back(filename);
	}

}
bool isexistcca(vector<string>& names, string& files)
{
	bool flag = 0;
	vector<string>::iterator it =find(names.begin(), names.end(), files);
	if (it != names.end())
		flag = 1;
	return flag;
		
}

void testcca(const string& modelpath, const string& pospath, const string& negpath, const string& resultpath)
{
	_mkdir(resultpath.c_str());
	vector<string> testfiles;
	Utils::findallfiles(pospath, testfiles, "png");
	sort(testfiles.begin(), testfiles.end());
	cout << "正样本样本数 " << testfiles.size() << endl;

	vector<string> negfiles;
	Utils::findallfiles(negpath, negfiles, "png");
	sort(negfiles.begin(), negfiles.end());

	cout << "负样本数 " << negfiles.size() << endl;

	string svmpath = modelpath + "cca_svm1.xml";
	string ccamatrixpath = modelpath + "ccamatrix.yml";
	CCAFUSION cca;
	cca.loadSvmDetector(svmpath);
	cca.loadccamatrix(ccamatrixpath);

	string output = resultpath + "cca_ori" + ".txt";
	vector<string> im_names;
	readcca(output, im_names);
	ofstream fout(output,ios::app);
//#pragma omp parallel for
	for (int i = 0; i < testfiles.size(); ++i)
	{    
		if (isexistcca(im_names, testfiles[i]))
		continue;
		cout << testfiles[i] << endl;
		string fullpath = pospath + testfiles[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
		if (sample.empty())
			continue;
		/*if (sample.rows != 128 || sample.cols != 64)
		{
		resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}*/
		vector<Rect> founds;
		vector<double> weights;

		cca.detectMultiScale(sample, founds, weights, 0.);
		Utils::NonMaximalSuppression2(founds, weights, 0.5, 0);
		//Mat imrgb;
		//sample.convertTo(imrgb, CV_8U, 255.0 / 8000);
		//cvtColor(imrgb, imrgb, CV_GRAY2BGR);

		for (int k = 0; k < founds.size(); ++k)
		{
			Rect r = founds[k];

			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > sample.cols)
				r.width = sample.cols - r.x;
			if (r.y + r.height > sample.rows)
				r.height = sample.rows - r.y;

			fout << testfiles[i] << " " << weights[k] << " "
				<< r.x << " " << r.y
				<< " " << r.width << " " << r.height << endl;

			//rectangle(imrgb, r, Scalar(0, 0, 255), 4);
		}
		//cv::imshow("test", imrgb);
		//cvWaitKey();
	}
//#pragma omp parallel for
	for (int i = 0; i < negfiles.size(); ++i)
	{
		cout << negfiles[i] << endl;

		string fullpath = negpath + negfiles[i];
		Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
		if (sample.empty())
			continue;

		vector<Rect> founds;
		vector<double> weights;

		cca.detectMultiScale(sample, founds, weights);
		Utils::NonMaximalSuppression2(founds, weights, 0.5, 0);
		for (int j = 0; j < founds.size(); ++j)
		{
			Rect r = founds[j];

			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > sample.cols)
				r.width = sample.cols - r.x;
			if (r.y + r.height > sample.rows)
				r.height = sample.rows - r.y;

			fout << negfiles[i] << " " << weights[j] << " "
				<< r.x << " " << r.y
				<< " " << r.width << " " << r.height << endl;
			//rectangle(imrgb, r, Scalar(0, 0, 255), 4);
		}
	}

	fout.close();
}

void testfrommultiDetectors(DetectionAlgorithm* detectors[], string detectornames[], const string modelpath, const string& testpath,
	const string& testnegpath, const string& resultpath, void(*pfunc)(Mat&, Mat&), string testtype)
{
	_mkdir(resultpath.c_str());
	vector<string> testfiles;
	Utils::findallfiles(testpath, testfiles, "png");
	sort(testfiles.begin(), testfiles.end());
	cout << "正样本样本数 " << testfiles.size() << endl;

	vector<string> negfiles;
	Utils::findallfiles(testnegpath, negfiles, "png");
	cout << "负样本数 " << negfiles.size() << endl;
	cout << testtype << endl;

	for (int j = 0; j < 9; ++j)
	{
		cout << detectornames[j] << endl;
		//#pragma omp parallel for
		for (int k = 0; k < 1; ++k)
		{
			string svmpath = modelpath + detectornames[j] + "_" + testtype;
			svmpath += (k == 0 ? "svm1.xml" : "svm2.xml");
			detectors[j]->loadSvmDetector(svmpath);

			string outputpath = resultpath + detectornames[j] + "_" + testtype;
			outputpath += (k == 0 ? "1.txt" : "2.txt");

			//输出结果
			fstream fout(outputpath, ios::out);
			double sumt = 0;
			for (int i = 0; i <testfiles.size(); i++)
			{
				cout << testfiles[i] << endl;
				string fullpath = testpath + testfiles[i];
				Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
				if (sample.empty())
					continue;
				/*if (sample.rows != 128 || sample.cols != 64)
				{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
				}*/
				Mat filterimg;
				if (pfunc == NULL)
				{
					filterimg = sample;
				}
				else
					pfunc(sample, filterimg);

				vector<Rect> founds;
				vector<double> weights;

				//double t1 = (double)getTickCount();
				detectors[j]->detectMultiScale(filterimg, founds, weights, 0.);
				Utils::NonMaximalSuppression2(founds, weights, 0.5, 0);
				//double t2 = (double)getTickCount();
				//sumt +=(t2-t1);
				/*Mat imrgb;
				filterimg.convertTo(imrgb, CV_8U, 255.0 / 8000);
				cvtColor(imrgb, imrgb, CV_GRAY2BGR);*/
				//imshow("ori", imrgb);
				//Mat imrgb2 = imrgb.clone();

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
					//if(weights[h]>=0.5)
					//rectangle(imrgb, r, Scalar(0, 0, 255), 4);
				}
				/*	imshow("test", imrgb);
				cvWaitKey(10);*/
			}
			//fout.close();
			//system("pause");
			//cout << sumt/getTickFrequency()<< endl;
			//负样本测试

			for (int i = 0; i < negfiles.size(); ++i)
			{
				cout << testpath[i] << endl;
				string fullpath = testnegpath + negfiles[i];
				Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
				if (sample.empty())
					continue;
				/*	if (sample.rows != 128 || sample.cols != 64)
				{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
				}*/
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
				Utils::NonMaximalSuppression2(founds, weights, 0.5, 0);
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
					//fout << negfiles[i] << " " << weights[h] << endl;
				}
			}

			fout.close();
		}
	}
}

int main()
{
	//testfromKinect();
	//testfromFiles();
	//waitKey();
	//testgp();
	//system("pause");
	//omp_set_num_threads(2);
	string testpath = "E:\\HumanData\\postestfull\\";
	string negpath = "E:\\HumanData\\negtestfull\\";
	string resultspath = "E:\\wangrun\\TestData\\result\\2\\";
	string modelpath = "E:\\wangrun\\TestData\\models\\1\\";
	string model_cca = "E:\\wangrun\\TestData\\models\\cca\\";


	DetectionAlgorithm* detectors[9];
	detectors[0] = new PLDPK();
	detectors[1] = new SLTP();
	detectors[2] = new LTDP();
	detectors[3] = new HONVNW();
	detectors[4] = new HDD();
	detectors[5] = new LDPK();
	//detectors[6] = new RDSF();
	detectors[6] = new MPLDPK();
	detectors[7] = new ELDP();
	detectors[8] = new PELDP();
	string detectornames[9] = { "PLDPK","SLTP","LTDP","HONV","HDD","LDPK","MPLDPK" ,"ELDP","PELDP" };

	//detectors[2]->set_signThreshold(70);
	//testfromKinect2(detectors[2], "F:\\liuhao\\testTrainSet\\models8\\PLDPK_MHsvm1.xml");

	//string testtypes[3] = {"ori","ori","ori"};
	//int k[3] = { 0,0,0 };
	//testgp(detectors, detectornames, modelpath, testpath, negpath, resultspath, testtypes,k);
	testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, NULL, "ori");
	//testcca(model_cca, testpath, negpath, resultspath);
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc1, "NN");
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc2, "MH");
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc4, "NNMH");
	//testfrommultiDetectors(detectors, detectornames, modelpath, testpath, negpath, resultspath, filterfunc3, "INPAINT");

	for (int i = 0; i <9; ++i)
	{
		delete detectors[i];
	}

	std::system("pause");
}
