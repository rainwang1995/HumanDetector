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
#include "LTDP.h"
#include "HDD.h"
#include "platt_scaling.h"
#include "RDSF.h"
#include "LDPK.h"
#include "PLDPK.h"
#include "ccafusion.h"
#include "ELDP.h"
#include "PELDP.h"
#include "MPLDPK.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

string hardpath;
string modelpath;
//获取训练样本路径
//bool getSamples(const string& path, vector<string>& filenames)
//{
//	ifstream sap(path.c_str());
//	if (!sap.is_open())
//	{
//		cerr << "sample open failed" << endl;
//		return false;
//	}
//
//	string filename;
//	while (getline(sap, filename) && !filename.empty())
//	{
//		filenames.push_back(filename);
//	}
//
//	return true;
//}

//训练融合特征分类器
void Traincca(const string& pospath,const string& negpath,const string& negoripath,string& modelpaths, const string& hardfilepath,
	const string& ccapath, void(*pfunc)(Mat&, Mat&), string traintype)
{
	_mkdir(modelpaths.c_str());
	vector<string> posfiles;
	vector<string> negfiles;
	vector<string> negorifiles;
	Utils::findallfiles(pospath, posfiles, "png");
	Utils::findallfiles(negpath, negfiles, "png");
	Utils::findallfiles(negoripath, negorifiles, "png");

	CCAFUSION cca;
	cca.loadccamatrix(ccapath);
	int featurelen = cca.getFeatureLen();
	
	Mat FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
	Mat LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);

	cout << "读取正样本" << endl;
	cout << "正样本数量：" << posfiles.size() << endl;

#pragma omp parallel for
//并行处理for循环
	for (int i = 0; i < posfiles.size(); ++i)
	{
		//cout << i << endl;
		string path = pospath + posfiles[i];
		//cout << posfiles[i] << endl;
		vector<float> description;
		description.reserve(featurelen);
		Mat sample = imread(path, IMREAD_ANYDEPTH);
		if (sample.rows != 128 || sample.cols != 64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}

		Mat filteredimg;
		filteredimg = sample;

		cca.compute(filteredimg, description);//计算 ldpk,hdd的特征

		float* ptr = FeatureMat.ptr<float>(i);
		memcpy(ptr, &description[0], sizeof(float)*featurelen);

		LabelMat.at<int>(i, 0) = 1;
	}
	cout << "正样本计算完毕" << endl;
	cout << "读取负样本" << endl;
	cout << "负样本数量：" << negfiles.size() << endl;

#pragma omp parallel for
	for (int i = 0; i < negfiles.size(); ++i)
	{
		//cout << i << endl;
		//cout << negfiles[i] << endl;
		string path = negpath + negfiles[i];
		vector<float> description;
		description.reserve(featurelen);
		Mat sample = imread(path, IMREAD_ANYDEPTH);
		if (sample.rows != 128 || sample.cols != 64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}

		Mat filteredimg;
		filteredimg = sample;
		//filterimg=sample;
		cca.compute(filteredimg, description);

		float* ptr = FeatureMat.ptr<float>(i + posfiles.size());
		memcpy(ptr, &description[0], sizeof(float)*featurelen);

		LabelMat.at<int>(i + posfiles.size(), 0) = -1;
	}
	
	/*FileStorage fs(featurename,FileStorage::WRITE);
	fs << "features" << FeatureMat;
	fs << "labels" << LabelMat;
	fs.release();*/

	//第一轮训练
	cout << "开始训练" << endl;
	Ptr<SVM> mysvm = SVM::create();
	mysvm->setKernel(SVM::LINEAR);
	mysvm->setType(SVM::C_SVC);
	mysvm->setC(1);
	mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
	//mysvm->trainAuto(tData, 5);
	mysvm->train(tData);
	mysvm->save(modelpaths + "cca_" + "svm1.xml");

	cout << "第一轮训练完毕" << endl;

	for (int h = 0; h < 1; ++h)
	{
		char hardName[256];

		sprintf(hardName, "%s%d\\", (hardfilepath + "cca"+ "\\" + traintype + "\\").c_str(), h);
		_mkdir((hardfilepath + "cca").c_str());
		_mkdir((hardfilepath + "cca"+ "\\" + traintype + "\\").c_str());
		_mkdir(hardName);
		string hardtemppath(hardName);
		int cursamplesize = FeatureMat.rows;
		cca.setSvmDetector(mysvm);
		for (int i = 0; i < negorifiles.size(); ++i)
		{
			//cout << i << endl;
			string path = negoripath + negorifiles[i];
			cout << path << endl;
			Mat sample = imread(path, IMREAD_ANYDEPTH);

			Mat filteredimg;
			if (pfunc == NULL)
			{
				filteredimg = sample;
			}
			else
				pfunc(sample, filteredimg);

			vector<Rect> found;
			vector<double> weights;
			cca.detectMultiScale(filteredimg, found, weights);

			for (int ii = 0; ii < found.size(); ++ii)
			{
				//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
				Rect r = found[ii];
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
				sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(), ii);//生成hard example图片的文件名  
				imwrite(saveName, hardExampleImg);//保存文件  
			}
			found.clear();
			sample.release();
		}

		vector<string> hardfiles;
		Utils::findallfiles(hardtemppath, hardfiles, "png");
		cout << "错误分类数: " << hardfiles.size() << endl;
		if (hardfiles.size() < 10)
		{
			break;
		}

		FeatureMat.resize(FeatureMat.rows + hardfiles.size());
		LabelMat.resize(LabelMat.rows + hardfiles.size());

#pragma omp parallel for
		for (int i = 0; i < hardfiles.size(); ++i)
		{
			string path = hardtemppath + hardfiles[i];
			cout << hardfiles[i] << endl;
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}
			//filterimg = sample;
			Mat filteredimg;
			if (pfunc == NULL)
			{
				filteredimg = sample;
			}
			else
				pfunc(sample, filteredimg);

			cca.compute(filteredimg, description);

			float* ptr = FeatureMat.ptr<float>(i + cursamplesize);
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i + cursamplesize, 0) = -1;
		}

		//train again
		cout << "再次训练: " << h << endl;
		tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
		//mysvm->trainAuto(tData);
		mysvm->train(tData);
		cout << "训练完毕" << endl;
		char svmname[256];
		sprintf(svmname, (modelpaths + "cca"+ "_" + traintype + "svm%d.xml").c_str(), h + 2);
		mysvm->save(svmname);
	}
}


//void savefeatures(DetectionAlgorithm* detectors[], string detectornames[], const string& pospath, const string& negpath, const string& negoripath,
//	const string& hardfilepath, const string& featurepaths,const string& modelpaths, void(*pfunc)(Mat&, Mat&), string traintype)
//{
//	_mkdir(hardfilepath.c_str());
//	_mkdir(featurepaths.c_str());
//	_mkdir(modelpaths.c_str());
//	vector<string> posfiles;
//	vector<string> negfiles;
//	vector<string> negorifiles;
//
//	Utils::findallfiles(pospath, posfiles, "png");
//	Utils::findallfiles(negpath, negfiles, "png");
//	Utils::findallfiles(negoripath, negorifiles, "png");
//
//
//	for (int j = 7; j < 8; ++j)
//	{
//		Mat FeatureMat, LabelMat;
//		int featurelen = detectors[j]->getFeatureLen();
//
//		FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
//		LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);
//
//		cout << "读取正样本" << endl;
//		cout << "正样本数量：" << posfiles.size() << endl;
//#pragma omp parallel for
//		for (int i = 0; i < posfiles.size(); ++i)
//		{
//			//cout << i << endl;
//			string path = pospath + posfiles[i];
//			//cout << posfiles[i] << endl;
//			vector<float> description;
//			description.reserve(featurelen);
//			Mat sample = imread(path, IMREAD_ANYDEPTH);
//			if (sample.rows != 128 || sample.cols != 64)
//			{
//				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
//			}
//
//			Mat filteredimg;
//			if (pfunc == NULL)
//			{
//				filteredimg = sample;
//			}
//			else
//				pfunc(sample, filteredimg);
//			//filterimg=sample;
//			detectors[j]->compute(filteredimg, description);
//
//			float* ptr = FeatureMat.ptr<float>(i);
//			memcpy(ptr, &description[0], sizeof(float)*featurelen);//将特征填到特征矩阵的对应行
//
//			LabelMat.at<int>(i, 0) = 1;
//		}
//		cout << "正样本计算完毕" << endl;
//		cout << "读取负样本" << endl;
//		cout << "负样本数量：" << negfiles.size() << endl;
//
//#pragma omp parallel for
//		for (int i = 0; i < negfiles.size(); ++i)
//		{
//			//cout << i << endl;
//			//cout << negfiles[i] << endl;
//			string path = negpath + negfiles[i];
//			vector<float> description;
//			description.reserve(featurelen);
//			Mat sample = imread(path, IMREAD_ANYDEPTH);
//			if (sample.rows != 128 || sample.cols != 64)
//			{
//				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
//			}
//
//			Mat filteredimg;
//			if (pfunc == NULL)
//			{
//				filteredimg = sample;
//			}
//			else
//				pfunc(sample, filteredimg);
//			//filterimg=sample;
//			detectors[j]->compute(filteredimg, description);
//
//			float* ptr = FeatureMat.ptr<float>(i + posfiles.size());
//			memcpy(ptr, &description[0], sizeof(float)*featurelen);
//
//			LabelMat.at<int>(i + posfiles.size(), 0) = -1;
//		}
//
//		////保存特征和label
//		//string featurename = featurepaths + detectornames[j] + "_" + traintype + "_feature.txt";
//		////string labelname = featurepaths + detectornames[j] + "_" + traintype + "_label.txt";
//		//ofstream ffeature(featurename);
//		////ofstream flabel(labelname);
//		//
//		//for (int rowindex = 0; rowindex < FeatureMat.rows;++rowindex)
//		//{
//		//	ffeature << LabelMat.at<int>(rowindex, 0) << " ";
//		//	for (int colindex = 0; colindex < FeatureMat.cols;++colindex)
//		//	{
//		//		ffeature << FeatureMat.at<float>(rowindex, colindex) << " ";
//		//	}
//		//	ffeature << endl;
//		//}
//		//ffeature.close();
//		/*FileStorage fs(featurename,FileStorage::WRITE);
//		fs << "features" << FeatureMat;
//		fs << "labels" << LabelMat;
//		fs.release();*/
//
//		//第一轮训练
//		cout << "开始训练" << endl;
//		Ptr<SVM> mysvm = SVM::create();
//		mysvm->setKernel(SVM::LINEAR);
//		//mysvm->setKernel(SVM::INTER);
//		//mysvm->setKernel(SVM::POLY);
//		//mysvm->setKernel(SVM::KernelTypes::RBF);
//		//mysvm->setKernel(SVM::SIGMOID);
//		mysvm->setType(SVM::C_SVC);
//		mysvm->setC(0.1);
//
//		//mysvm->setDegree(3);
//		mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 100, 1e-6));
//		Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
//		double t = (double)getTickCount();
//		mysvm->trainAuto(tData,5);		
//		mysvm->train(tData);
//		double t2 = (double)getTickCount();
//		cout << (t2 - t) / getTickFrequency()<<endl;
//		mysvm->save(modelpaths + detectornames[j] + "_" + traintype + "svm1.xml");
//
//		cout << "第一轮训练完毕，boosstrap" << endl;
//		//计算platt_scaling参数
////#pragma region platt_scaleing
////		detectors[j]->setSvmDetector(mysvm);
////		Mat preLabel(LabelMat.size(), CV_32SC1);
////		preLabel.setTo(-1);
////		cout << "platt_scaling" << endl;
////		for (int i = 0; i < posfiles.size(); ++i)
////		{
////			string path = pospath + posfiles[i];
////			//cout << posfiles[i] << endl;
////			vector<float> description;
////			description.reserve(featurelen);
////			Mat sample = imread(path, IMREAD_ANYDEPTH);
////			if (sample.rows != 128 || sample.cols != 64)
////			{
////				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
////			}
////
////			Mat filteredimg;
////			if (pfunc == NULL)
////			{
////				filteredimg = sample;
////			}
////			else
////				pfunc(sample, filteredimg);
////			//filterimg=sample;
////			vector<Rect> founds;
////			vector<double> weights;
////			detectors[j]->detectMultiScale(filteredimg, founds, weights);
////			if (!founds.empty())
////			{
////				preLabel.at<int>(i, 0) = 1;
////			}
////		}
////
////		for (int i = 0; i < negfiles.size(); ++i)
////		{
////			string path = negpath + negfiles[i];
////			//cout << posfiles[i] << endl;
////			vector<float> description;
////			description.reserve(featurelen);
////			Mat sample = imread(path, IMREAD_ANYDEPTH);
////			if (sample.rows != 128 || sample.cols != 64)
////			{
////				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
////			}
////
////			Mat filteredimg;
////			if (pfunc == NULL)
////			{
////				filteredimg = sample;
////			}
////			else
////				pfunc(sample, filteredimg);
////			//filterimg=sample;
////			vector<Rect> founds;
////			vector<double> weights;
////			detectors[j]->detectMultiScale(filteredimg, founds, weights);
////			if (!founds.empty())
////			{
////				preLabel.at<int>(i + posfiles.size(), 0) = 1;
////			}
////		}
////
////		PlattScaling ps;
////		ps.sigmoidTrain(preLabel, LabelMat);
////		ps.save(modelpaths + detectornames[j] + "_" + traintype + "sigma1.xml");
////#pragma endregion
//
//		
//		//bootstrap
//		for (int h = 0; h < 0; ++h)
//		{
//			char hardName[256];
//
//			sprintf(hardName, "%s%d\\", (hardfilepath + detectornames[j] + "\\" + traintype + "\\").c_str(), h);
//			_mkdir((hardfilepath + detectornames[j]).c_str());
//			_mkdir((hardfilepath + detectornames[j] + "\\" + traintype + "\\").c_str());
//			_mkdir(hardName);
//			string hardtemppath(hardName);
//			int cursamplesize = FeatureMat.rows;
//			detectors[j]->setSvmDetector(mysvm);
//#pragma omp parallel for
//			for (int i = 0; i < negorifiles.size(); ++i)
//			{
//				//cout << i << endl;
//				string path = negoripath + negorifiles[i];
//				Mat sample = imread(path, IMREAD_ANYDEPTH);
//
//				Mat filteredimg;
//				if (pfunc == NULL)
//				{
//					filteredimg = sample;
//				}
//				else
//					pfunc(sample, filteredimg);
//
//				vector<Rect> found;
//				vector<double> weights;
//				detectors[j]->detectMultiScale(filteredimg, found, weights);
//
//				for (int j = 0; j < found.size(); ++j)
//				{
//					//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
//					Rect r = found[j];
//					if (r.x < 0)
//						r.x = 0;
//					if (r.y < 0)
//						r.y = 0;
//					if (r.x + r.width > sample.cols)
//						r.width = sample.cols - r.x;
//					if (r.y + r.height > sample.rows)
//						r.height = sample.rows - r.y;
//
//					//将矩形框保存为图片，就是Hard Example  
//					Mat hardExampleImg = sample(r);//从原图上截取矩形框大小的图片
//					char saveName[256];//裁剪出来的负样本图片文件名
//					string hardsavepath = hardtemppath + negorifiles[i];
//					hardsavepath.erase(hardsavepath.end() - 4, hardsavepath.end());
//					resize(hardExampleImg, hardExampleImg, Size(64, 128), INTER_NEAREST);//将剪裁出来的图片缩放为64*128大小  
//					sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(), j);//生成hard example图片的文件名  
//					imwrite(saveName, hardExampleImg);//保存文件  
//				}
//				found.clear();
//				sample.release();
//			}
//
//			vector<string> hardfiles;
//			Utils::findallfiles(hardtemppath, hardfiles, "png");
//			cout << "错误分类数: " << hardfiles.size() << endl;
//			if (hardfiles.size() < 10)
//			{
//				break;
//			}
//
//			FeatureMat.resize(FeatureMat.rows + hardfiles.size());
//			LabelMat.resize(LabelMat.rows + hardfiles.size());
//
//#pragma omp parallel for
//			for (int i = 0; i < hardfiles.size(); ++i)
//			{
//				string path = hardtemppath + hardfiles[i];
//				//cout << hardfiles[i] << endl;
//				vector<float> description;
//				description.reserve(featurelen);
//				Mat sample = imread(path, IMREAD_ANYDEPTH);
//				if (sample.rows != 128 || sample.cols != 64)
//				{
//					resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
//				}
//				//filterimg = sample;
//				Mat filteredimg;
//				if (pfunc == NULL)
//				{
//					filteredimg = sample;
//				}
//				else
//					pfunc(sample, filteredimg);
//
//				detectors[j]->compute(filteredimg, description);
//
//				float* ptr = FeatureMat.ptr<float>(i + cursamplesize);
//				memcpy(ptr, &description[0], sizeof(float)*featurelen);
//
//				LabelMat.at<int>(i + cursamplesize, 0) = -1;
//			}
//
//			//train again
//			cout << "再次训练: " << h << endl;
//			tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
//			//mysvm->trainAuto(tData,5);
//			
//			mysvm->train(tData);
//			cout << "训练完毕" << endl;
//			char svmname[256];
//			sprintf(svmname, (modelpaths + detectornames[j] + "_" + traintype + "svm%d.xml").c_str(), h + 2);
//			mysvm->save(svmname);
//
////#pragma region platt_scaleing
////			detectors[j]->setSvmDetector(mysvm);
////
////			Mat preLabel(LabelMat.size(), CV_32SC1);
////			preLabel.setTo(-1);
////			cout << "platt_scaling" << endl;
////			for (int i = 0; i < posfiles.size(); ++i)
////			{
////				string path = pospath + posfiles[i];
////				//cout << posfiles[i] << endl;
////				vector<float> description;
////				description.reserve(featurelen);
////				Mat sample = imread(path, IMREAD_ANYDEPTH);
////				if (sample.rows != 128 || sample.cols != 64)
////				{
////					resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
////				}
////
////				Mat filteredimg;
////				if (pfunc == NULL)
////				{
////					filteredimg = sample;
////				}
////				else
////					pfunc(sample, filteredimg);
////				//filterimg=sample;
////				vector<Rect> founds;
////				vector<double> weights;
////				detectors[j]->detectMultiScale(filteredimg, founds, weights);
////				if (!founds.empty())
////				{
////					preLabel.at<int>(i, 0) = 1;
////				}
////			}
////
////			for (int i = 0; i < negfiles.size(); ++i)
////			{
////				string path = negpath + negfiles[i];
////				//cout << posfiles[i] << endl;
////				vector<float> description;
////				description.reserve(featurelen);
////				Mat sample = imread(path, IMREAD_ANYDEPTH);
////				if (sample.rows != 128 || sample.cols != 64)
////				{
////					resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
////				}
////
////				Mat filteredimg;
////				if (pfunc == NULL)
////				{
////					filteredimg = sample;
////				}
////				else
////					pfunc(sample, filteredimg);
////				//filterimg=sample;
////				vector<Rect> founds;
////				vector<double> weights;
////				detectors[j]->detectMultiScale(filteredimg, founds, weights);
////				if (!founds.empty())
////				{
////					preLabel.at<int>(i + posfiles.size(), 0) = 1;
////				}
////			}
////
////			PlattScaling ps;
////			ps.sigmoidTrain(preLabel, LabelMat);
////			ps.save(modelpaths + detectornames[j] + "_" + traintype + "sigma2.xml");
////#pragma endregion
//		}
//
//		/*string featurename = featurepaths + detectornames[j] + "_" + traintype + "_feature.xml";
//		FileStorage fs(featurename,FileStorage::WRITE);
//		fs << "features" << FeatureMat;
//		fs << "labels" << LabelMat;
//		fs.release();*/
//	}
//}

void Train(DetectionAlgorithm* detectors[], string detectornames[], const string& pospath, const string& negpath, const string& negoripath,
	const string& hardfilepath, const string& modelpaths, void(*pfunc)(Mat&, Mat&), string traintype)
{
	_mkdir(hardfilepath.c_str());
	_mkdir(modelpaths.c_str());
	vector<string> posfiles;
	vector<string> negfiles;
	vector<string> negorifiles;

	Utils::findallfiles(pospath, posfiles,"png");
	Utils::findallfiles(negpath, negfiles,"png");
	Utils::findallfiles(negoripath, negorifiles, "png");

	//string detectornames[4] = { "SLTP","HDD","LTDP","HONV" };
//#pragma omp parallel for 
	for (int j = 0; j < 10; ++j)
	{
		Ptr<SVM> mysvm = SVM::create();
		mysvm->setKernel(SVM::LINEAR);
		mysvm->setType(SVM::C_SVC);
		mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

		Mat FeatureMat, LabelMat;
		int featurelen = detectors[j]->getFeatureLen();

		FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
		LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);

		cout << "读取正样本" << endl;
		cout << "正样本数量：" << posfiles.size() << endl;
#pragma omp parallel for
		for (int i = 0; i < posfiles.size(); ++i)
		{
			//cout << i << endl;
			string path = pospath + posfiles[i];
			cout << posfiles[i] << endl;
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}

			Mat filteredimg;
			if (pfunc==NULL)
			{
				filteredimg = sample;
			}
			else
				pfunc(sample, filteredimg);
			//filterimg=sample;
			detectors[j]->compute(filteredimg, description);

			float* ptr = FeatureMat.ptr<float>(i);
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i, 0) = 1;
		}
		cout << "正样本计算完毕" << endl;
		cout << "读取负样本" << endl;
		cout << "负样本数量：" << negfiles.size() << endl;

#pragma omp parallel for
		for (int i = 0; i < negfiles.size(); ++i)
		{
			//cout << i << endl;
			cout << negfiles[i] << endl;
			string path = negpath + negfiles[i];
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}

			Mat filteredimg;
			if (pfunc == NULL)
			{
				filteredimg = sample;
			}
			else
				pfunc(sample, filteredimg);
			//filterimg=sample;
			detectors[j]->compute(filteredimg, description);

			float* ptr = FeatureMat.ptr<float>(i + posfiles.size());
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i + posfiles.size(), 0) = -1;
		}

		//第一轮训练
		cout << "开始训练" << endl;
		Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
		//mysvm->trainAuto(tData);
		mysvm->train(tData);
		mysvm->save(modelpaths + detectornames[j] + "_"+traintype+"svm1.xml");
		cout << "第一轮训练完毕，boosstrap" << endl;

		//计算platt_scaling参数
		//Mat preLabel(LabelMat.size(), CV_32SC1);
		//preLabel.setTo(-1);
		//cout << "platt_scaling" << endl;
		//for (int i = 0; i < posfiles.size();++i)
		//{
		//	string path = pospath + posfiles[i];
		//	//cout << posfiles[i] << endl;
		//	vector<float> description;
		//	description.reserve(featurelen);
		//	Mat sample = imread(path, IMREAD_ANYDEPTH);
		//	if (sample.rows != 128 || sample.cols != 64)
		//	{
		//		resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		//	}

		//	Mat filteredimg;
		//	if (pfunc == NULL)
		//	{
		//		filteredimg = sample;
		//	}
		//	else
		//		pfunc(sample, filteredimg);
		//	//filterimg=sample;
		//	vector<Rect> founds;
		//	vector<double> weights;
		//	detectors[j]->detectMultiScale(filteredimg, founds, weights);
		//	if (!founds.empty())
		//	{
		//		preLabel.at<int>(i, 0) = 1;
		//	}
		//}

		//for (int i = 0; i < negfiles.size(); ++i)
		//{
		//	string path = negpath + negfiles[i];
		//	//cout << posfiles[i] << endl;
		//	vector<float> description;
		//	description.reserve(featurelen);
		//	Mat sample = imread(path, IMREAD_ANYDEPTH);
		//	if (sample.rows != 128 || sample.cols != 64)
		//	{
		//		resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		//	}

		//	Mat filteredimg;
		//	if (pfunc == NULL)
		//	{
		//		filteredimg = sample;
		//	}
		//	else
		//		pfunc(sample, filteredimg);
		//	//filterimg=sample;
		//	vector<Rect> founds;
		//	vector<double> weights;
		//	detectors[j]->detectMultiScale(filteredimg, founds, weights);
		//	if (!founds.empty())
		//	{
		//		preLabel.at<int>(i+posfiles.size(), 0) = 1;
		//	}
		//}
		//PlattScaling ps;
		//ps.sigmoidTrain(preLabel, LabelMat);
		//ps.save(modelpaths + detectornames[j] + "_" + traintype + "sigma1.xml");
		//bootstrap
		for (int h = 0; h < 1; ++h)
		{
			char hardName[256];

			sprintf(hardName, "%s%d\\", (hardfilepath+detectornames[j]+"\\"+traintype+"\\").c_str(), h);
			_mkdir((hardfilepath + detectornames[j]).c_str());
			_mkdir((hardfilepath + detectornames[j] + "\\" + traintype + "\\").c_str());
			_mkdir(hardName);
			string hardtemppath(hardName);
			int cursamplesize = FeatureMat.rows;
			detectors[j]->setSvmDetector(mysvm);		
			for (int i = 0; i < negorifiles.size(); ++i)
			{
				//cout << i << endl;
				string path = negoripath + negorifiles[i];
				Mat sample = imread(path, IMREAD_ANYDEPTH);

				Mat filteredimg;
				if (pfunc == NULL)
				{
					filteredimg = sample;
				}
				else
					pfunc(sample, filteredimg);

				vector<Rect> found;
				vector<double> weights;
				detectors[j]->detectMultiScale(filteredimg, found, weights);

				for (int ii = 0; ii < found.size(); ++ii)
				{
					//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
					Rect r = found[ii];
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
					sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(),ii);//生成hard example图片的文件名  
					imwrite(saveName, hardExampleImg);//保存文件  
				}
				found.clear();
				sample.release();
			}

			vector<string> hardfiles;
			Utils::findallfiles(hardtemppath, hardfiles, "png");
			cout << "错误分类数: " << hardfiles.size() << endl;
			if (hardfiles.size()<10)
			{
				break;
			}

			FeatureMat.resize(FeatureMat.rows + hardfiles.size());
			LabelMat.resize(LabelMat.rows + hardfiles.size());

#pragma omp parallel for
			for (int i = 0; i < hardfiles.size(); ++i)
			{
				string path = hardtemppath + hardfiles[i];
				//cout << hardfiles[i] << endl;
				vector<float> description;
				description.reserve(featurelen);
				Mat sample = imread(path, IMREAD_ANYDEPTH);
				if (sample.rows != 128 || sample.cols != 64)
				{
					resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
				}
				//filterimg = sample;
				Mat filteredimg;
				if (pfunc == NULL)
				{
					filteredimg = sample;
				}
				else
					pfunc(sample, filteredimg);

				detectors[j]->compute(filteredimg, description);

				float* ptr = FeatureMat.ptr<float>(i + cursamplesize);
				memcpy(ptr, &description[0], sizeof(float)*featurelen);

				LabelMat.at<int>(i + cursamplesize, 0) = -1;
			}

			//train again
			cout << "再次训练: " << h << endl;
			tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
			//mysvm->trainAuto(tData);
			mysvm->train(tData);
			cout << "训练完毕" << endl;
			char svmname[256];
			sprintf(svmname, (modelpaths + detectornames[j] + "_" + traintype+ "svm%d.xml").c_str(), h + 2);
			mysvm->save(svmname);
		}
	}
}

//训练单个分类器
//void Train(DetectionAlgorithm* detector,const string& pospath, const vector<string>& posfiles, 
//	const string& negpath, const vector<string>& negfiles, const string& negoripath, const vector<string>& negorifiles)
//{
//	Ptr<SVM> mysvm = SVM::create();
//	mysvm->setKernel(SVM::LINEAR);
//	mysvm->setType(SVM::C_SVC);
//	mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//
//	Mat FeatureMat;
//	Mat LabelMat;
//
//	int featurelen = detector->getFeatureLen();
//	FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
//	LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);
//
//	//读取正样本
//	cout << "读取正样本" << endl;
//	cout << "正样本数量：" << posfiles.size() << endl;
//#pragma omp parallel for default(none) shared(detector,FeatureMat,featurelen,LabelMat,pospath,posfiles)
//	for (int i = 0; i < posfiles.size();++i)
//	{
//		string path = pospath + posfiles[i];
//		//cout << posfiles[i] << endl;
//		vector<float> description;
//		description.reserve(featurelen);
//		Mat sample = imread(path,IMREAD_ANYDEPTH);
//		if (sample.rows!= 128 ||sample.cols!=64)
//		{
//			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
//		}
//		Mat filterimg;
//		preProcessing::pixelFilter(sample, filterimg);
//		Mat painted;
//		preProcessing::inpaintrawdepth(filterimg, painted, 3, 0.5);
//
//		//filterimg=sample;
//		detector->compute(painted, description);
//
//		float* ptr = FeatureMat.ptr<float>(i);
//		memcpy(ptr, &description[0], sizeof(float)*featurelen);
//
//		LabelMat.at<int>(i, 0) = 1;
//	}
//
//	cout << "正样本计算完毕" << endl;
//	cout << "读取负样本" << endl;
//	cout << "负样本数量：" << negfiles.size() << endl;
//
//#pragma omp parallel for default(none) shared(detector,FeatureMat,featurelen,LabelMat,negpath,negfiles,posfiles)
//	for (int i = 0; i < negfiles.size(); ++i)
//	{
//		//cout << negfiles[i] << endl;
//		string path = negpath + negfiles[i];
//		vector<float> description;
//		description.reserve(featurelen);
//		Mat sample = imread(path, IMREAD_ANYDEPTH);
//		if (sample.rows != 128 || sample.cols != 64)
//		{
//			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
//		}
//		Mat filterimg;
//		
//		preProcessing::pixelFilter(sample, filterimg);
//		Mat painted;
//		preProcessing::inpaintrawdepth(filterimg, painted,3,0.5);
//		//filterimg=sample;
//		detector->compute(painted, description);
//
//		float* ptr = FeatureMat.ptr<float>(i+posfiles.size());
//		memcpy(ptr, &description[0], sizeof(float)*featurelen);
//
//		LabelMat.at<int>(i+posfiles.size(), 0) = -1;
//	}
//
//	//设置训练数据
//	cout << "开始训练" << endl;
//	Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
//	//mysvm->trainAuto(tData);
//	mysvm->train(tData);
//	cout << "第一轮训练完毕" << endl;
//	//在原始负样本上测试
//	mysvm->save((modelpath + "sltpsvmfm1.xml").c_str());
//	
//	for (int h = 0; h < 2;++h)
//	{
//		char hardName[256];
//		sprintf(hardName, "%s%d\\", hardpath.c_str(), h);
//		_mkdir(hardName);
//		string hardtemppath(hardName);
//
//		detector->setSvmDetector(mysvm);
//		int cursamplesize = FeatureMat.rows;
//		//cout << cursamplesize << endl;
//
//#pragma omp parallel for		
//		for (int i = 0; i < negorifiles.size(); ++i)
//		{
//			//cout << i << endl;
//			string path = negoripath + negorifiles[i];
//			Mat sample = imread(path, IMREAD_ANYDEPTH);
//			Mat filterimg;
//			preProcessing::pixelFilter(sample, filterimg);
//			//filterimg = sample;
//			vector<Rect> found;
//			vector<double> weights;
//			vector<Point> ts;
//			//honv.detect(sample, ts, weights);
//			detector->detectMultiScale(filterimg, found, weights);
//			/*if (found.size() > 0)
//				cout << found.size() << endl;*/
//		    for (int j = 0; j < found.size(); ++j)
//			{
//				//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
//				Rect r = found[j];
//				if (r.x < 0)
//					r.x = 0;
//				if (r.y < 0)
//					r.y = 0;
//				if (r.x + r.width > sample.cols)
//					r.width = sample.cols - r.x;
//				if (r.y + r.height > sample.rows)
//					r.height = sample.rows - r.y;
//
//				//将矩形框保存为图片，就是Hard Example  
//				Mat hardExampleImg = sample(r);//从原图上截取矩形框大小的图片
//				char saveName[256];//裁剪出来的负样本图片文件名
//				string hardsavepath = hardtemppath + negorifiles[i];
//				hardsavepath.erase(hardsavepath.end() - 4, hardsavepath.end());
//				resize(hardExampleImg, hardExampleImg, Size(64, 128), INTER_NEAREST);//将剪裁出来的图片缩放为64*128大小  
//				sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(), j);//生成hard example图片的文件名  
//				imwrite(saveName, hardExampleImg);//保存文件  
//			}
//			found.clear();
//			sample.release();
//		}
//
//		vector<string> hardfiles;
//		Utils::findallfiles(hardtemppath, hardfiles, "png");
//		cout << "错误分类数: " << hardfiles.size() << endl;
//		if (hardfiles.size()<10)
//		{
//			break;
//		}
//		
//		FeatureMat.resize(FeatureMat.rows + hardfiles.size());
//		LabelMat.resize(LabelMat.rows + hardfiles.size());
//
//#pragma omp parallel for
//		for (int i = 0; i < hardfiles.size();++i)
//		{
//			string path = hardtemppath + hardfiles[i];
//			//cout << hardfiles[i] << endl;
//			vector<float> description;
//			description.reserve(featurelen);
//			Mat sample = imread(path, IMREAD_ANYDEPTH);
//			if (sample.rows != 128 || sample.cols != 64)
//			{
//				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
//			}
//			Mat filterimg;
//			preProcessing::pixelFilter(sample, filterimg);
//			//filterimg = sample;
//			detector->compute(filterimg, description);
//
//			float* ptr = FeatureMat.ptr<float>(i+ cursamplesize);
//			memcpy(ptr, &description[0], sizeof(float)*featurelen);
//
//			LabelMat.at<int>(i+ cursamplesize, 0) = -1;
//		}
//
//		//train again
//		cout << "再次训练: " <<h<< endl;
//		tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
//		//mysvm->trainAuto(tData);
//		mysvm->train(tData);
//		cout << "训练完毕" << endl;
//		char svmname[256];
//		sprintf(svmname, (modelpath+"sltpsvmfm1%d.xml").c_str(), h+2);
//		mysvm->save(svmname);
//	}
//	//提取hardfiles的特征
//	//mysvm->save("sltpsvm2.xml");
//	cout << "训练完毕" << endl;
//}

int main()
{
	omp_set_num_threads(2);

	string negpath = "E:\\HumanData\\negtrain2\\";
	string pospath = "E:\HumanData\\postrain2\\";

	string negoripath = "E:\HumanData\\negtrainfull\\";
	
	hardpath = "E:\\wangrun\\TestData\\hardfiles2\\";
	modelpath = "E:\\wangrun\\TestData\\models\\";
	string ccapath = "E:\\HumanData\\models\\ccamatrix.yml";

	DetectionAlgorithm* detectors[10];

	detectors[0] = new SLTP();
	detectors[1] = new PLDPK();
	detectors[2] = new LTDP();
	detectors[3] = new HONVNW();
	detectors[4] = new RDSF();
	detectors[5] = new HDD();
	detectors[6] = new LDPK();
	detectors[7] = new MPLDPK();
	detectors[8] = new ELDP();
	detectors[9] = new PELDP();
	string detectornames[10] = { "SLTP","PLDPK","LTDP","HONV","RDSF","HDD","LDPK","MPLDPK" ,"ELDP","PELDP"};
	
	//savefeatures(detectors, detectornames, pospath, negpath, negoripath, hardpath, featurepath, modelpath, NULL, "ori");
	//savefeatures(detectors, detectornames, pospath, negpath, negoripath, hardpath, featurepath, modelpath, filterfunc1, "NN");
	//savefeatures(detectors, detectornames, pospath, negpath, negoripath, hardpath, featurepath, modelpath, filterfunc2, "MH");
	//savefeatures(detectors, detectornames, pospath, negpath, negoripath, hardpath, featurepath, modelpath, filterfunc4, "NNMH");
	//savefeatures(detectors, detectornames, pospath, negpath, negoripath, hardpath, featurepath, modelpath, filterfunc3, "INPAINT");

	//detectors[2]->set_signThreshold(70);
	//Train(detectors, detectornames, pospath, negpath, negoripath, hardpath, modelpath, NULL,"ori");
	//Train(detectors, detectornames, pospath, negpath, negoripath, hardpath, modelpath, filterfunc1, "NN");
	//Train(detectors, detectornames, pospath, negpath, negoripath, hardpath, modelpath, filterfunc2, "MH");
	//Train(detectors, detectornames, pospath, negpath, negoripath, hardpath, modelpath, filterfunc3, "INPAINT");
	//Train(detectors, detectornames, pospath, negpath, negoripath, hardpath, modelpath, filterfunc4, "NNMH");
	Traincca(pospath, negpath, negoripath, modelpath, hardpath, ccapath, NULL, "ori");

	for (int i = 0; i < 10;++i)
	{
		delete detectors[i];
	}

	system("pause");
}
