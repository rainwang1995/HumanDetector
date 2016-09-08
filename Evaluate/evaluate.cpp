#include "evaluate.h"
#include <fstream>
#include <strstream>
#include <iterator>
#include "Utils.h"
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
		double weight = 0.0;
		int x, y, width, height;
		istr >> filename >> x >> y >> width >> height;
		filename = filename + ".png";
		/*if (filename[0] == 'i')
		{
			filename = filename + ".png";
		}
		else
		{
			sprintf(name, "img_%04s.png", filename.c_str());
			filename = string(name);
		}*/
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

void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps)
{
	if (groupThreshold <= 0 || rectList.empty() || rectList.size() == 1)
	{
		return;
	}

	CV_Assert(rectList.size() == weights.size());

	vector<int> labels;
	int nclasses = partition(rectList, labels, SimilarRects(eps));

	vector<cv::Rect_<double> > rrects(nclasses);
	vector<int> numInClass(nclasses, 0);
	vector<double> foundWeights(nclasses, -std::numeric_limits<double>::max());
	int i, j, nlabels = (int)labels.size();

	for (i = 0; i < nlabels; i++)
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		foundWeights[cls] = max(foundWeights[cls], weights[i]);
		numInClass[cls]++;
	}

	for (i = 0; i < nclasses; i++)
	{
		// find the average of all ROI in the cluster
		cv::Rect_<double> r = rrects[i];
		double s = 1.0 / numInClass[i];
		rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
			cv::saturate_cast<double>(r.y*s),
			cv::saturate_cast<double>(r.width*s),
			cv::saturate_cast<double>(r.height*s));
	}

	rectList.clear();
	weights.clear();

	for (i = 0; i < nclasses; i++)
	{
		cv::Rect r1 = rrects[i];
		int n1 = numInClass[i];
		double w1 = foundWeights[i];
		if (n1 <= groupThreshold)
			continue;
		// filter out small rectangles inside large rectangles
		for (j = 0; j < nclasses; j++)
		{
			int n2 = numInClass[j];

			if (j == i || n2 <= groupThreshold)
				continue;

			cv::Rect r2 = rrects[j];

			int dx = cv::saturate_cast<int>(r2.width * eps);
			int dy = cv::saturate_cast<int>(r2.height * eps);

			if (r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3))
				break;
		}

		if (j == nclasses)
		{
			rectList.push_back(r1);
			weights.push_back(w1);
		}
	}
}

void readresults(const string& path, mapresults& results, vector<double>& scores)
{
	ifstream readfile(path);
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
		double weight = 0.0;
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
	for (; itr != results.end(); itr++)
	{
		/*std::sort(itr->second.begin(), itr->second.end(), [](const DetectionResult& a, const DetectionResult& b) {
		return a.weight > b.weight;
		});*/
		vector<DetectionResult>::iterator citr = itr->second.begin();
		for (; citr != itr->second.end(); citr++)
		{
			double score_ = citr->weight;
			//cout << score_ << endl;
			scores.push_back(score_);
		}
	}
	sort(scores.begin(), scores.end());
}

void readfppw(const string& path, mapresults& results)
{
	ifstream readfile(path);
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
		double weight = 0.0;

		istr >> filename >> weight;
		weight = 1 / (1 + exp(-weight));
		Rect r(0, 0, 64, 128);

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
}

void filterresult(const mapresults& results, mapresults& filtered, double thr)
{
	filtered.clear();
	mapresults::const_iterator itr = results.cbegin();
	for (; itr != results.cend(); ++itr)
	{
		vector<Rect> locations;
		vector<double> weights;
		vector<DetectionResult>::const_iterator citr = itr->second.cbegin();
		for (; citr != itr->second.cend(); ++citr)
		{
			if (citr->weight >= thr)
			{
				locations.push_back(citr->box);
				weights.push_back(citr->weight);
			}
		}
		if (locations.empty())
			continue;
		//Utils::NonMaximalSuppression(locations, weights, 0.5, 0);
		//groupRectangles(locations, weights);
		filtered[itr->first] = vector<DetectionResult>(locations.size());
		for (int i = 0; i < locations.size(); ++i)
		{
			DetectionResult temp;
			temp.box = locations[i];
			temp.weight = weights[i];
			filtered[itr->first][i] = temp;
		}
	}
}

Missratefppi computemissratefppi(mapresults & annotation, mapresults & results)
{
	int fp = 0, fn = 0;
	int tp = 0, tn = 0;

	//标志groundtruth是否已经被匹配
	map<string, vector<bool> > flags;
	for (mapresults::iterator it = annotation.begin(); it != annotation.end(); it++)
	{
		flags[it->first] = vector<bool>(it->second.size(), false);
	}

	mapresults::iterator itr = results.begin();
	for (; itr != results.end(); ++itr)
	{
		//如果在annotation中不存在，fp直接加
		if (annotation.count(itr->first) == 0)
		{
			fp += (itr->second.size());
			//for (int i = 0; i < itr->second.size();++i)
			//{
			//	if (itr->second[i].weight>=hitThreshold)
			//	{
			//		++fp;
			//	}
			//}
			continue;
		}
		string filename = itr->first;
		vector<DetectionResult>& temp = itr->second;

		//目前每个样本只有一个annotation，先简化
		for (int j = 0; j < annotation[filename].size(); ++j)
		{
			if (flags[filename][j] == false)
			{
				for (int i = 0; i < temp.size(); ++i)
				{
					//if (temp[i].weight<hitThreshold)
					//{
					//	//fp = fp + (temp.size() - i);
					//	break;
					//}
					float overlapr = computeoverlap(temp[i].box, annotation[filename][j].box);
					if (overlapr >= 0.5)
					{
						++tp;
						for (int k = i + 1; k < temp.size(); ++k)
						{
							fp++;
						}
						flags[filename][j] = true;
						break;
					}
					fp++;
				}
			}
		}
	}

	//	ofstream fout("fkcca.txt");
	//统计annotation中未匹配的
	map<string, vector<bool> >::iterator pitr = flags.begin();
	for (; pitr != flags.end(); ++pitr)
	{
		for (int i = 0; i < pitr->second.size(); ++i)
		{
			if (pitr->second[i] == false)
			{
				//				fout << pitr->first << endl;
				++fn;
			}
		}
	}
	//	fout.close();
	flags.clear();

	Missratefppi missrate;
	missrate.missrate = fn;
	missrate.fppi = fp;
	return missrate;
}

Missratefppi computemissratefppw(mapresults& annotation, mapresults& results)
{
	int fp = 0, fn = 0;
	int tp = 0, tn = 0;

	//标志groundtruth是否已经被匹配
	map<string, vector<bool> >flags;
	for (mapresults::iterator it = annotation.begin(); it != annotation.end(); ++it)
	{
		flags[it->first] = vector<bool>(it->second.size(), false);
	}

	mapresults::iterator itr = results.begin();
	for (; itr != results.end(); ++itr)
	{
		//如果在annotation中不存在，fp直接加
		if (annotation.count(itr->first) == 0)
		{
			fp += (itr->second.size());
			//for (int i = 0; i < itr->second.size();++i)
			//{
			//	if (itr->second[i].weight>=hitThreshold)
			//	{
			//		++fp;
			//	}
			//}
			continue;
		}
		string filename = itr->first;
		vector<DetectionResult> temp = itr->second;

		//目前每个样本只有一个annotation，先简化
		for (int j = 0; j < annotation[filename].size(); ++j)
		{
			if (flags[filename][j] == false)
			{
				for (int i = 0; i < temp.size(); ++i)
				{
					++tp;
					flags[filename][j] = true;
				}
			}
		}
	}

	//统计annotation中未匹配的
	map<string, vector<bool> >::iterator pitr = flags.begin();
	for (; pitr != flags.end(); ++pitr)
	{
		for (int i = 0; i < pitr->second.size(); ++i)
		{
			if (pitr->second[i] == false)
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

//Missratefppi computemissratefppi(const string& path1, const string& path2, const string& path3,float hitThreshold)
//{
//	mapresults annotation,results;
//	readannotation(path1, annotation);
//	readresults(path2,path3, results);
//
//	return computemissratefppi(annotation, results, hitThreshold);
//}

float computeoverlap(cv::Rect a, cv::Rect b)
{
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}
