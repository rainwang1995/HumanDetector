#include "evaluate.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace cv;
using namespace std;

int cntfilelines(const string& filepath)
{
	int cnt = 0;
	fstream filein(filepath);
	if (filein.is_open())
	{
		string txtline;
		while (getline(filein, txtline));
		{
			if (!txtline.empty())
			{
				++cnt;
			}
		}
	}
	return cnt;
}

int main()
{
	string apath = "F:\\liuhao\\testTrainSet\\annotation.txt";
	string posresult = "F:\\liuhao\\testTrainSet\\reslut\\hddpos1.txt";
	string negresult = "F:\\liuhao\\testTrainSet\\reslut\\hddneg1.txt";

	//Missratefppi missrate = computemissratefppi(apath, posresult, 0);
	//int fp = cntfilelines(negresult);
	int posnum = 3312;
	int negnum = 2133;
	string outfile = "F:\\liuhao\\testTrainSet\\reslut\\missratehdd1.xls";
	string outfiletxt= "F:\\liuhao\\testTrainSet\\reslut\\missratehdd1.txt";
	ofstream fout(outfile);
	ofstream fouttxt(outfiletxt);
	mapresults results;
	mapresults annotations;
	readannotation(apath, annotations);
	readresults(posresult, negresult, results);

	for (float th = 0.0; th <=1; th += 0.05)
	{
		Missratefppi missrate = computemissratefppi(annotations, results, th);

		cout << "miss rate: " << missrate.missrate / (double)(posnum) << endl;
		cout << "fppi: " << missrate.fppi / (double)(posnum + negnum) << endl;
		cout << endl;

		missrate.missrate /= (posnum);
		missrate.fppi /= (posnum + negnum);
		fout << missrate.missrate<< "\t";
		fout << missrate.fppi<<"\t";
		fout << endl;

		fouttxt << missrate.missrate << " ";
		fouttxt << missrate.fppi << endl;
	}
	fout.close();
	fouttxt.close();
	system("pause");
}