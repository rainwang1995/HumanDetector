#include "evaluate.h"
#include <fstream>
#include <strstream>
#include <iterator>
using namespace cv;

void readannotation(const string & path, mapresults & annotation)
{
	fstream readfile(path, ios::in);
	if (!readfile.is_open())
	{
		cerr << "error file" << endl;
		return;
	}
	char name[256];
	string txtline;
	while (getline(readfile, txtline))
	{
		istringstream istr(txtline);

		string filename;
		float weight = 0.0;
		int x, y, width, height;
		istr >> filename >> x >> y >> width >> height;
		sprintf(name, "img_%04s.png", filename.c_str());
		filename = string(name);
		Rect r(x, y, width, height);

		DetectionResult dresult(r, weight);
		if (!annotation.count(filename))
		{
			annotation[filename] = vector<DetectionResult>();
			annotation[filename].push_back(dresult);
		}
		else
			annotation[filename].push_back(dresult);
	}

	readfile.close();
}

void readresults(const string& path, const string& path2,mapresults& results)
{
	fstream readfile(path, ios::in);
	if (!readfile.is_open())
	{
		cerr << "error file" << endl;
		return;
	}

	string txtline;
	while (getline(readfile, txtline))
	{
		istringstream istr(txtline);

		string filename;
		float weight = 0.0;
		int x, y, width, height;
		istr >> filename >>weight>> x >> y >> width >> height;
		weight = 1 / (1 + exp(-weight));
		Rect r(x, y, width, height);

		DetectionResult dresult(r, weight);
		if (!results.count(filename))
		{
			results[filename] = vector<DetectionResult>();
			results[filename].push_back(dresult);
		}
		else
			results[filename].push_back(dresult);
	}
	readfile.close();
	readfile.open(path2);
	while (getline(readfile, txtline))
	{
		istringstream istr(txtline);

		string filename;
		float weight = 0.0;
		int x, y, width, height;
		istr >> filename >> weight >> x >> y >> width >> height;

		weight = 1 / (1 + exp(-weight));

		Rect r(x, y, width, height);

		DetectionResult dresult(r, weight);
		if (!results.count(filename))
		{
			results[filename] = vector<DetectionResult>();
			results[filename].push_back(dresult);
		}
		else
			results[filename].push_back(dresult);
	}
	readfile.close();
	//按weight排序,由大到小
	mapresults::iterator itr = results.begin();
	for (; itr != results.end();++itr)
	{
		std::sort(itr->second.begin(), itr->second.end(), [](const DetectionResult& a, const DetectionResult& b) {
			return a.weight > b.weight;
		});
	}
}

Missratefppi computemissratefppi(mapresults & annotation, mapresults & results, float hitThreshold)
{
	int fp = 0,fn=0;
	int tp = 0, tn = 0;

	//标志groundtruth是否已经被匹配
	map<string, vector<bool> >flags;
	for (mapresults::iterator it = annotation.begin(); it != annotation.end();++it)
	{
		flags[it->first] = vector<bool>(it->second.size(),false);
	}

	mapresults::iterator itr = results.begin();
	for (; itr != results.end();++itr)
	{
		//如果在annotation中不存在，fp直接加
		if (annotation.count(itr->first)==0)
		{
			for (int i = 0; i < itr->second.size();++i)
			{
				if (itr->second[i].weight>=hitThreshold)
				{
					++fp;
				}
			}
			continue;
		}
		string filename = itr->first;
		vector<DetectionResult> temp = itr->second;

		//目前每个样本只有一个annotation，先简化
		for (int j = 0; j < annotation[filename].size();++j)
		{
			if (flags[filename][j]==false)
			{
				for (int i = 0; i < temp.size();++i)
				{
					if (temp[i].weight<hitThreshold)
					{
						//fp = fp + (temp.size() - i);
						break;
					}
					float overlapr = computeoverlap(temp[i].box, annotation[filename][j].box);
					if (overlapr>=0.5)
					{
						++tp;
						for (int k = i+1; k < temp.size();++k)
						{
							if (temp[k].weight>=hitThreshold)
							{
								++fp;
							}
						}

						flags[filename][j] = true;
						break;
					}
				}
			}
		}

		for (int j = 0; j < annotation[filename].size();++j)
		{
			if (flags[filename][j] == false)
			{
				++fn;
			}
		}

	}
		
	Missratefppi missrate;
	missrate.missrate = fn;
	missrate.fppi = fp;
	return missrate;
}

Missratefppi computemissratefppi(const string& path1, const string& path2, const string& path3,float hitThreshold)
{
	mapresults annotation,results;
	readannotation(path1, annotation);
	readresults(path2,path3, results);

	return computemissratefppi(annotation, results, hitThreshold);
}

float computeoverlap(cv::Rect a, cv::Rect b)
{
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}
