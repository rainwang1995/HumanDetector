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

class GraphProgation
{
public:
	GraphProgation():theta(0),eta(0.1),detectornums(4) { setParameters(); }
	GraphProgation(float _theta, float _eta) :theta(_theta), eta(_eta),detectornums(4) { setParameters(); }

	~GraphProgation()
	{
		for (int i = 0; i < 4;++i)
		{
			delete detectors[i];
		}
	}
public:
	void detect(const cv::Mat& src, vector<cv::Rect>& foundloactions, vector<double>& outputweights)const;
	void setSvmDetectors(const vector<string>& paths);
	
private:
	DetectionAlgorithm* detectors[4];
	float theta;
	float eta;
	const int detectornums;
private:
	void setParameters();

public:
	void mergeRects(const vector<cv::Rect>& candidates,
		const vector<double>& weights, vector<cv::Rect>& output, vector<double> outputweight)const;
	void computeRelationMatrix(const vector<cv::Rect>& candidates, cv::Mat& relations)const;
};


#endif
