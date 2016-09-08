#ifndef GRAPHPROPAGATION_H
#define GRAPHPROPAGATION_H

#include <iostream>
#include <vector>
#include "opencvHeader.h"
#include "Algorithm.hpp"
#include "HDD.h"
#include "LTDP.h"
#include "SLTP.h"
#include "HONVNW.h"
#include "preprocess.h"
#include "PLDPK.h"
#include "LDPK.h"
class GraphProgation
{
public:
	//GraphProgation():theta(0),eta(0.1),detectornums(4),setdetector(false) { 
	//	//setParameters(); 
	//}

	GraphProgation(int _detectornums) :detectornums(_detectornums) {
		detectors = new DetectionAlgorithm*[detectornums];
		setdetector = true;
	}
	GraphProgation(float _theta, float _eta) :theta(_theta), eta(_eta),detectornums(4),setdetector(false) { 
		//setParameters();
	}

	~GraphProgation()
	{
		if (setdetector)
		{
			for (int i = 0; i < detectornums; ++i)
			{
				if (detectors[i] != NULL)
				{
					delete detectors[i];
				}
			}
			delete[] detectors;
		}
		
	}
public:
	void detect(const cv::Mat& src, vector<cv::Rect>& foundloactions, vector<double>& outputweights)const;
	void detect(const cv::Mat& src, vector<cv::Rect>& foundlocations, vector<double>& outputweights, string svmtypes[])const;
	void setSvmDetectors(const vector<string>& paths);
	void setSvmDetectors(DetectionAlgorithm* detectors[]);

private:

	DetectionAlgorithm** detectors;
	float theta;
	float eta;
	const int detectornums;
private:
	//void setParameters();
	bool setdetector;
public:
	void mergeRects(const vector<cv::Rect>& candidates,
		const vector<double>& weights, vector<cv::Rect>& output, vector<double>& outputweight)const;
	void computeRelationMatrix(const vector<cv::Rect>& candidates, cv::Mat& relations)const;
};


#endif
