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
string hardpath;

bool getSamples(const string& path, vector<string>& filenames)
{
	ifstream sap(path.c_str());
	if (!sap.is_open())
	{
		cerr << "sample open failed" << endl;
		return false;
	}

	string filename;
	while (getline(sap, filename) && !filename.empty())
	{
		filenames.push_back(filename);
	}

	return true;
}

void honvTrain(const string& pospath, const vector<string>& posfiles, const string& negpath, const vector<string>& negfiles, const string& negoripath, const vector<string>& negorifiles)
{
	HONVNW honv;
	Ptr<SVM> mysvm = SVM::create();
	mysvm->setKernel(SVM::LINEAR);
	mysvm->setType(SVM::C_SVC);
	mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	Mat FeatureMat;
	Mat LabelMat;

	int featurelen = honv.getFeatureLen();
	FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
	LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);

	//读取正样本
	cout << "读取正样本" << endl;
#pragma omp parallel for default(none) shared(honv,FeatureMat,featurelen,LabelMat,pospath,posfiles)
	for (int i = 0; i < posfiles.size();++i)
	{
		string path = pospath + posfiles[i];
		cout << i << endl;
		vector<float> description;
		description.reserve(featurelen);
		Mat sample = imread(path,IMREAD_ANYDEPTH);
		if (sample.rows!= 128 ||sample.cols!=64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}
		Mat filterimg;
		preProcessing::pixelFilter(sample, filterimg);

		honv.compute(filterimg, description);

		float* ptr = FeatureMat.ptr<float>(i);
		memcpy(ptr, &description[0], sizeof(float)*featurelen);

		LabelMat.at<int>(i, 0) = 1;
	}

	cout << "正样本计算完毕" << endl;
	cout << "读取负样本" << endl;
#pragma omp parallel for default(none) shared(honv,FeatureMat,featurelen,LabelMat,negpath,negfiles,posfiles)
	for (int i = 0; i < negfiles.size(); ++i)
	{
		cout << i << endl;
		string path = negpath + negfiles[i];
		vector<float> description;
		description.reserve(featurelen);
		Mat sample = imread(path, IMREAD_ANYDEPTH);
		if (sample.rows != 128 || sample.cols != 64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}
		Mat filterimg;
		preProcessing::pixelFilter(sample, filterimg);

		honv.compute(filterimg, description);

		float* ptr = FeatureMat.ptr<float>(i+posfiles.size());
		memcpy(ptr, &description[0], sizeof(float)*featurelen);

		LabelMat.at<int>(i+posfiles.size(), 0) = -1;
	}

	//设置训练数据
	cout << "开始训练" << endl;
	Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
	mysvm->train(tData);
	cout << "第一轮训练完毕" << endl;
	//在原始负样本上测试
	
	
	for (int h = 0; h < 3;++h)
	{
		char hardName[256];
		sprintf(hardName, "%s%d\\", hardpath.c_str(), h);
		_mkdir(hardName);
		string hardtemppath(hardName);

		honv.setSvmDetector(mysvm);

#pragma omp parallel for		
		for (int i = 0; i < negorifiles.size(); ++i)
		{
			string path = negoripath + negorifiles[i];
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			vector<Rect> found;
			vector<double> weights;
			vector<Point> ts;
			//honv.detect(sample, ts, weights);
			honv.detectMultiScale(sample, found, weights);
			for (int j = 0; j < found.size(); ++j)
			{
				//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
				Rect r = found[j];
				if (r.x < 0)
					r.x = 0;
				if (r.y < 0)
					r.y = 0;
				if (r.x + r.width > sample.cols)
					r.width = sample.cols - r.x;
				if (r.y + r.height > sample.rows)
					r.height = sample.rows - r.y;

				//将矩形框保存为图片，就是Hard Example  
				Mat hardExampleImg = sample(r);//从原图上截取矩形框大小的图片
				char saveName[256];//裁剪出来的负样本图片文件名
				string hardsavepath = hardtemppath + negorifiles[i];
				hardsavepath.erase(hardsavepath.end() - 4, hardsavepath.end());
				resize(hardExampleImg, hardExampleImg, Size(64, 128), INTER_NEAREST);//将剪裁出来的图片缩放为64*128大小  
				sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(), j);//生成hard example图片的文件名  
				imwrite(saveName, hardExampleImg);//保存文件  
			}
		}

		vector<string> hardfiles;
		Utils::findallfiles(hardtemppath, hardfiles, "png");
		if (hardfiles.size()<10)
		{
			break;
		}
		cout << "错误分类数: "<<hardfiles.size() << endl;

		FeatureMat.resize(FeatureMat.rows + hardfiles.size());

		LabelMat.resize(LabelMat.rows + hardfiles.size());

#pragma omp parallel for
		for (int i = 0; i < hardfiles.size();++i)
		{
			string path = hardtemppath + hardfiles[i];
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}
			Mat filterimg;
			preProcessing::pixelFilter(sample, filterimg);

			honv.compute(filterimg, description);

			float* ptr = FeatureMat.ptr<float>(i+negfiles.size() + posfiles.size());
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i+ negfiles.size() + posfiles.size(), 0) = -1;
		}

		//train again
		cout << "再次训练" << endl;
		tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
		mysvm->train(tData);
	}
	

	//提取hardfiles的特征

	mysvm->save("svm.xml");
}

int main()
{
	//string negsaps = "F:\\liuhao\\INRIAPerson\\INRIAPerson\\negphoto\\neg.txt";
	//string possaps = "F:\\liuhao\\INRIAPerson\\INRIAPerson\\train_64x128_H96\\pos\\pos.txt";

	string negpath = "F:\\liuhao\\testTrainSet\\neg\\";
	string pospath = "F:\\liuhao\\testTrainSet\\pos\\";

	string negoripath = "F:\\liuhao\\testTrainSet\\negori\\";
	
	hardpath = "F:\\liuhao\\testTrainSet\\hardimages\\";
	_mkdir(hardpath.c_str());
	
	vector<string> posfiles;
	vector<string> negfiles;

	vector<string> negorifiles;

	Utils::findallfiles(pospath, posfiles, "png");
	Utils::findallfiles(negpath, negfiles, "png");
	Utils::findallfiles(negoripath, negorifiles, "png");

	honvTrain(pospath, posfiles, negpath, negfiles, negoripath, negorifiles);
}
