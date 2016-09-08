#include "evaluate.h"
#include <iostream>
#include <fstream>
#include <string>
#include "Utils.h"
#include <io.h>
#include <direct.h>

using namespace cv;
using namespace std;

//int cntfilelines(const string& filepath)
//{
//	int cnt = 0;
//	fstream filein(filepath);
//	if (filein.is_open())
//	{
//		string txtline;
//		while (getline(filein, txtline));
//		{
//			if (!txtline.empty())
//			{
//				++cnt;
//			}
//		}
//	}
//	return cnt;
//}
//
//void testfppw()
//{
//	string apath = "F:\\liuhao\\testTrainSet\\testannotation.txt";
//	string result = "F:\\liuhao\\testTrainSet\\fppw\\";
//
//	vector<string> resultsfile;
//	Utils::findallfiles(result, resultsfile, "txt");
//
//	//Missratefppi missrate = computemissratefppi(apath, posresult, 0);
//	//int fp = cntfilelines(negresult);
//	int posnum = 3312;
//	int negnum = 106650;
//
//	string outputpath = "F:\\liuhao\\testTrainSet\\fppwresults\\";
//	_mkdir(outputpath.c_str());
//
//	mapresults annotations;
//	readannotation(apath, annotations);
//	//readresults(result, results);
//#pragma omp parallel for
//	for (int i = 0; i < resultsfile.size(); ++i)
//	{
//		mapresults results;
//		mapresults filtered;
//		size_t pos2 = resultsfile[i].find(".");
//		string svmname = resultsfile[i].substr(0, pos2);
//		readfppw(result + resultsfile[i], results);
//
//		size_t pos = resultsfile[i].find("_");
//
//		string outfile = outputpath + svmname + ".xls";
//		string outfiletxt = outputpath + svmname + ".txt";
//
//		ofstream fout(outfile);
//		ofstream fouttxt(outfiletxt);
//
//		for (float th = 0.0; th <= 1;)
//		{
//			filterresult(results, filtered, th);
//			Missratefppi missrate = computemissratefppw(annotations, filtered);
//
//			//cout << "miss rate: " << missrate.missrate / (double)(posnum) << endl;
//			//cout << "fppi: " << missrate.fppi / (double)(posnum + negnum) << endl;
//			//cout << endl;
//
//			missrate.missrate /= (posnum);
//			missrate.fppi /= (posnum + negnum);
//			fout << missrate.missrate << "\t";
//			fout << missrate.fppi << "\t";
//			fout << endl;
//
//			fouttxt << missrate.missrate << " ";
//			fouttxt << missrate.fppi << endl;
//
//			if (th < 0.5)
//			{
//				th += 0.1;
//			}
//			else
//				th += 0.005;
//
//		}
//		fout.close();
//		fouttxt.close();
//		results.clear();
//		filtered.clear();
//	}
//}

int main()
{

	string apath = "E:\\HumanData2\\testannotation.txt";
	string result = "E:\\wangrun\\TestData2\\result\\2\\";

	vector<string> resultsfile;
	Utils::findallfiles(result, resultsfile, "txt");

	//Missratefppi missrate = computemissratefppi(apath, posresult, 0);
	//int fp = cntfilelines(negresult);
	int posnum = 3706;
	int negnum = 3235;

	string outputpath = "E:\\wangrun\\TestData\\evaluationdata\\1\\";
	_mkdir(outputpath.c_str());
	mapresults annotations;
	readannotation(apath, annotations);
	//readresults(result, results);
//#pragma omp parallel for
	for (int i =0; i <resultsfile.size(); ++i)
	{
		cout << resultsfile[i] << endl;
		mapresults results;
		mapresults filtered;
		vector<double> scores;
		size_t pos2 = resultsfile[i].find(".");
		string svmname = resultsfile[i].substr(0, pos2);
		readresults(result + resultsfile[i], results, scores);
		string outfile = outputpath + svmname + ".xls";
		//string outfiletxt = outputpath + svmname + ".txt";
		ofstream fout(outfile);
		//ofstream fouttxt(outfiletxt);
		for (double th =0.5; th <=1;)
		{
			filterresult(results, filtered, th);
			Missratefppi missrate = computemissratefppi(annotations, filtered);

			//cout << "miss rate: " << missrate.missrate / (double)(posnum) << endl;
			//cout << "fppi: " << missrate.fppi / (double)(posnum + negnum) << endl;
			//cout << endl;

			missrate.missrate /= (posnum);
			missrate.fppi /= (posnum + negnum);
			fout << missrate.missrate << "\t";
			fout << missrate.fppi << "\t";
			fout << endl;

			/*	fouttxt << missrate.missrate << " ";
				fouttxt << missrate.fppi << endl;*/

			if (th < 0.5)
			{
				th += 0.01;
			}
			else
				th += 0.005;

		}

		//for (vector<double>::iterator p = scores.begin(); p != scores.end(); p++)
		//{
		//	double th = *p;
		//	filterresult(results, filtered, th);
		//	Missratefppi missrate = computemissratefppi(annotations, filtered);
		//	missrate.missrate /= (posnum);
		//	missrate.fppi /= (posnum + negnum);
		//		fout << missrate.missrate << "\t";
		//		fout << missrate.fppi << "\t";
		//		fout << endl;
		//	//fouttxt << missrate.missrate << " ";
		//	//fouttxt << missrate.fppi << endl;
		//    cout << missrate.missrate << " " << missrate.fppi << endl;
		//}
		fout.close();
		//fouttxt.close();
		results.clear();
		filtered.clear();
		cout << "µÚ" << i << "ÂÖ½áÊø£¡" << endl;
	}
	system("pause");
}