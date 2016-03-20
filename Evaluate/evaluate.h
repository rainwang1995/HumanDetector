#ifndef EVALUATE_H
#define EVALUATE_H
#include "opencvHeader.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

struct DetectionResult 
{
public:
	DetectionResult()
	{
		box = cv::Rect();
		weight = 0;
	}
	DetectionResult(cv::Rect _box,float _w):box(_box),weight(_w){}
	cv::Rect box;
	float weight;
};

struct Missratefppi
{
	float missrate;
	float fppi;
};

typedef map<string, vector<DetectionResult> > mapresults;

void readannotation(const string& path, mapresults& annotation);

void readresults(const string& path,const string& path2, mapresults& results);

Missratefppi computemissratefppi(mapresults& annotation, mapresults& results, float hitThreshold);

Missratefppi computemissratefppi(const string& path1, const string& path2,const string& path3, float hitThreshold);

float computeoverlap(cv::Rect a, cv::Rect b);
#endif
