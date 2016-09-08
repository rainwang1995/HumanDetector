#include "Graphpropagation.h"
#include <omp.h>
#include <cmath>

using namespace cv;

void GraphProgation::detect(const cv::Mat& src, vector<cv::Rect>& foundloactions,vector<double>& outputweights)const
{
	vector<vector<Rect> > founds(detectornums);
	vector<vector<double> > weights(detectornums);

#pragma omp parallel for
	for (int i = 0; i < detectornums;++i)
	{
		detectors[i]->detectMultiScale(src, founds[i], weights[i],0.0);

		if (founds[i].size()>1)
		{
			//按权重排序,插入排序
			for (int j = 1; j < founds[i].size(); ++j)
			{
				int k = j - 1;
				cv::Rect rt = founds[i][j];
				double   tempweight = weights[i][j];
				while (k >= 0 && weights[i][k] < tempweight)
				{
					founds[i][k + 1] = founds[i][k];
					weights[i][k + 1] = weights[i][k + 1];
					--k;
				}

				weights[i][k + 1] = tempweight;
				founds[i][k + 1] = rt;
			}
		}		
	}

	//每个分类器选择权重最大框
	vector<Rect> maxweightRects(detectornums);
	vector<double> maxWeights(detectornums);
	for (int i = 0; i < maxWeights.size();++i)
	{
		if (founds[i].empty())
		{
			maxweightRects[i] = Rect();
			maxWeights[i] = 0;
			continue;
		}
		maxweightRects[i] = founds[i][0];
		maxWeights[i] = 1 / (1 + exp(-weights[i][0]));
	}

	mergeRects(maxweightRects, maxWeights, foundloactions,outputweights);
}

void GraphProgation::detect(const cv::Mat& src, vector<cv::Rect>& foundlocations, vector<double>& outputweights, string svmtypes[]) const
{
	vector<vector<Rect> > founds(detectornums);
	vector<vector<double> > weights(detectornums);

#pragma omp parallel for
	for (int i = 0; i < detectornums; ++i)
	{
		Mat temp = src;
		Mat filterimg;
		if (svmtypes[i] == "ori")
			filterimg = src;
		else if (svmtypes[i]=="MH")
		{
			filterfunc2(temp, filterimg);
		}
		else if (svmtypes[i]=="NN")
		{
			filterfunc1(temp, filterimg);
		}
		else if (svmtypes[i] == "NNMH")
		{
			filterfunc4(temp, filterimg);
		}
		else
			filterfunc3(temp, filterimg);

		detectors[i]->detectMultiScale(filterimg, founds[i], weights[i], 0.);

		if (founds[i].size()>1)
		{
			//按权重排序,插入排序
			for (int j = 1; j < founds[i].size(); ++j)
			{
				int k = j - 1;
				cv::Rect rt = founds[i][j];
				double   tempweight = weights[i][j];
				while (k >= 0 && weights[i][k] < tempweight)
				{
					founds[i][k + 1] = founds[i][k];
					weights[i][k + 1] = weights[i][k + 1];
					--k;
				}

				weights[i][k + 1] = tempweight;
				founds[i][k + 1] = rt;
			}
		}
	}

	//每个分类器选择权重最大框
	vector<Rect> maxweightRects(detectornums);
	vector<double> maxWeights(detectornums);
	for (int i = 0; i < maxWeights.size(); ++i)
	{
		if (founds[i].empty())
		{
			maxweightRects[i] = Rect();
			maxWeights[i] = 0;
			continue;
		}
		maxweightRects[i] = founds[i][0];
		maxWeights[i] = 1 / (1 + exp(-weights[i][0]));
	}

	mergeRects(maxweightRects, maxWeights, foundlocations, outputweights);
}

void GraphProgation::setSvmDetectors(const vector<string>& paths)
{

	if (paths.size()!= detectornums)
	{
		cerr << "error" << endl;
		return;
	}

	//setParameters();
	//detectors[0]->loadSvmDetector(paths[0]);

	//DetectionAlgorithm* sltp = new SLTP();
	//sltp->loadSvmDetector(paths[3]);

	for (int i = 0; i < paths.size();++i)
	{
		detectors[i]->loadSvmDetector(paths[i]);
	}
}

void GraphProgation::setSvmDetectors(DetectionAlgorithm* _detectors[])
{
	for (int i = 0; i < detectornums; i++)
	{
		detectors[i] = _detectors[i];
	}
}


void GraphProgation::mergeRects(const vector<cv::Rect>& candidates, const vector<double>& weights, vector<cv::Rect>& output,vector<double>& outputweight) const
{
	Mat relations;
	computeRelationMatrix(candidates, relations);
	//cout << relations << endl;
	vector<double> fweight(weights.begin(), weights.end());

	Mat Cs(1, detectornums, CV_64FC1, (void*)&fweight[0]);
	Mat C0 = Cs.clone();

	Mat THETA = Mat::zeros(1, detectornums, CV_64FC1);
	THETA.at<double>(0) = 0.33;
	THETA.at<double>(1) = 0.33;
	THETA.at<double>(2) = 0.4;

	//Mat THETA = (Mat_<double>(1, 3) << (0.33, 0.33, 0.4));

	Mat Cs1(1, detectornums, CV_64FC1);
	Mat DELTA = (relations)/(detectornums-1);

	for (int i = 0; i < DELTA.rows;++i)
	{
		DELTA.row(i) *= (1 - THETA.at<double>(i));
	}

	DELTA = DELTA.t();
	//DELTA = abs(THETA - 1)*DELTA;
	int itr = 0;
	int MAX_INTERATOR = 100;
	int cnt=0;
	while (true)
	{
		//cout << ++itr << endl;
		Cs1 = Cs*DELTA + THETA.mul(C0);
		//cout << "S1:" << endl;
		//cout<< Cs1 << endl;
		//cout << "S:" << endl;
		//cout << Cs << endl;
		//计算差距
		Mat diff = (Cs1 - Cs);
		diff = diff*diff.t();
		++cnt;
		//cout << sqrt(diff.at<double>(0, 0)) << endl;
		if (sqrt(diff.at<double>(0,0))<=1e-6||cnt>=MAX_INTERATOR)
		{
			break;
		}

		Cs1.copyTo(Cs);
	}

	//计算高斯混合模型权重
	Rect2f finalRect;

	//Rect finalRect=Rect();
	double sumCf = sum(Cs1)[0];
	if (abs(sumCf-0.0)<1e-9)
	{
		return;
	}
	finalRect = Rect2f();
	double weight = 0.0;
	for (int i = 0; i < detectornums;++i)
	{
		double wi = Cs1.at<double>(i)/sumCf;
		finalRect.x += (wi*(candidates[i].x+candidates[i].width/2));
		finalRect.y += (wi*(candidates[i].y+candidates[i].height/2));
		finalRect.width += (wi*candidates[i].width);
		finalRect.height += (wi*candidates[i].height);

		weight += (wi*weights[i]);
	}

	output.push_back(Rect(round(finalRect.x-finalRect.width/2),round(finalRect.y-finalRect.height/2),
		round(finalRect.width),round(finalRect.height)));
	outputweight.push_back(weight);
}

void GraphProgation::computeRelationMatrix(const vector<cv::Rect>& candidates, cv::Mat& relations) const
{
	relations=Mat::zeros(detectornums, detectornums, CV_64FC1);

	//Mat Pr=Mat::zeros(4, 4, CV_32FC1);
	//Mat Re=Mat::zeros(4, 4, CV_32FC1);

	for (int i = 0; i < detectornums;++i)
	{
		for (int j = 0; j < detectornums;++j)
		{
			if (i==j)
			{
				continue;
			}
			
			if (candidates[i].area()==0||candidates[j].area()==0)
			{
				relations.at<double>(i, j) = 0;
				continue;
			}
			double pr = 1.0*(candidates[i] & candidates[j]).area() / candidates[j].area();
			double re = 1.0*(candidates[i] & candidates[j]).area() / candidates[i].area();

			if (abs(pr-0.0)<1e-6||abs(re-0.0)<1e-6)
			{
				relations.at<double>(i, j) = 0;
				continue;
			}
			relations.at<double>(i, j) = 2 * pr*re / (pr + re);
		}
	}	
}
