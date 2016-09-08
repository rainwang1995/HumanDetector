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
	DetectionResult(cv::Rect _box,double _w):box(_box),weight(_w){}
	cv::Rect box;
	double weight;
};

struct Missratefppi
{
	float missrate;
	float fppi;
};

typedef map<string, vector<DetectionResult> > mapresults;

void readannotation(const string& path, mapresults& annotation);

void readresults(const string& path, mapresults& results, vector<double>& scores);

void readfppw(const string& path, mapresults& results);

void filterresult(const mapresults& results, mapresults& filtered, double thr);

Missratefppi computemissratefppi(mapresults& annotation, mapresults& results);

Missratefppi computemissratefppw(mapresults& annotation, mapresults& results);

//Missratefppi computemissratefppi(const string& path1, const string& path2,const string& path3, float hitThreshold);

float computeoverlap(cv::Rect a, cv::Rect b);

void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold=2, double eps=0.2);
#endif
