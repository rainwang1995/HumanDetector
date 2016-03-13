#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <fstream>
#include "opencvHeader.h"
#include "Utils.h"
#include <cstdlib>
#include <ctime>
#include <string>
#include <io.h>
#include <direct.h>
#include <iterator>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

int CropImageCount = 0; //裁剪出来的负样本图片个数

void raw2depth(vector<float>& rawdata)
{
	for (vector<float>::iterator it = rawdata.begin(); it != rawdata.end();++it)
	{
		//float val;
		if (*it < 2047)
		{
			*it = 0.1236*tanf(*it / 2842.5 + 1.1863);
			*it *= 1000;
		}
		else
			*it = 0;

		//depth.push_back(val);
	}
}

void sampleScale(Mat src,string& imgpath,string& imgname)
{
	char saveName[256];
	if (src.cols >= 64 && src.rows >= 128)
	{
		srand(time(NULL));//设置随机数种子
		for (int j = 0; j < 10; j++)
		{
			int x = (rand() % (src.cols - 64)); //x坐标
			int y = (rand() % (src.rows - 128)); //y坐标

			string filename = imgname;
			filename.erase(filename.end() - 4, filename.end());
			Mat imgROI = src(Rect(x, y, 64, 128));
			string imagepath = imgpath + filename + "_%03d.png";
			sprintf(saveName, imagepath.c_str(), j);//生成裁剪出的负样本图片的文件名
			imwrite(saveName, imgROI);//保存文件


			Mat imgflip;//反转
			flip(imgROI, imgflip, 1);
			sprintf(saveName, (imgpath + filename + "_%03d.png").c_str(), j + 10);//生成裁剪出的负样本图片的文件名
			imwrite(saveName, imgflip);//保存文件
		}
	}
}

int main(int argc, char* argv[])
{
	string negpath = "F:\\liuhao\\testTrainSet\\negori\\";
	string savepath = "F:\\liuhao\\testTrainSet\\neg2\\";
	_mkdir(savepath.c_str());
	vector<string> files;
	if (argc==3)
	{
		negpath = argv[1];
		savepath = argv[2];
	}
	Utils::findallfiles(negpath, files, "png");
	sort(files.begin(), files.end());
	//把yml转换成png
//#pragma omp parallel for
//	for (int i = 0; i < files.size();++i)
//	{
//		string ImgName = negpath + files[i];
//		cout << files[i] << endl;
//		FileStorage fr(ImgName,FileStorage::READ);
//		//fr.open(ImgName, FileStorage::READ);
//		if (fr.isOpened())
//		{
//			FileNode Node = fr.getFirstTopLevelNode();
//
//			FileNode dataNode = Node["data"];
//			vector<float> data;
//
//			if (dataNode.isSeq())
//			{
//				//FileNodeIterator it = dataNode.begin(), it_end = dataNode.end();
//				/*for (; it != it_end;++it)
//				{
//					if (i>=63)
//					{
//						cout << (float)*it;
//					}
//					data.push_back((float)*it);
//				}*/
//				dataNode >> data;
//			}
//			//cout << Node.name()<<endl;
//			if (Node.name()=="depth")
//			{
//				raw2depth(data);
//			}
//			else
//			{
//				for (int j = 0; j < data.size();++j)
//				{
//					data[j] *= 1000;
//				}
//			}
//			//Mat img=Mat_<float>(480, 640);
//			Mat img(480, 640, CV_32FC1,&data[0]);
//
//			//Mat img16;
//			//img.convertTo(img16, CV_16UC1,&data[0]);
//			img.convertTo(img, CV_16UC1);
//			string imgname = files[i];
//			imgname.erase(imgname.end() - 4, imgname.end());
//			//sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//生成裁剪出的负样本图片的文件名
//			imwrite(savepath+ imgname+".png", img);//保存文
//			//cvReleaseImage(&img);
//		}	
//	}
	int nlevels = 64;
	float scale0 = 1.05;

#pragma omp parallel for
	for (int i = 0; i < files.size();++i)
	{
		char saveName[256];//裁剪出来的负样本图片文件名
		cout << "处理：" << files[i] << endl;
		string ImgName = negpath + files[i];

		Mat src = imread(ImgName, IMREAD_ANYDEPTH);//读取图片

		vector<float> scales;
		float scale = 1;
		for (int l = 0; l < nlevels;++l)
		{
			scales.push_back(scale);
				if (cvRound(src.cols / scale) < 64 || cvRound(src.rows / scale) < 128
					|| scale0 <= 1)
				{
					break;
				}

				scale *= scale0;	
		}
		 

		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//设置随机数种子
        //从每张图片中随机裁剪10个64*128大小的不包含人的负样本
			for (int j = 0; j < 10; j++)
			{
				int x = (rand() % (src.cols - 64)); //x坐标
				int y = (rand() % (src.rows - 128)); //y坐标
				
				string filename = files[i];
				filename.erase(filename.end() - 4, filename.end());
				Mat imgROI = src(Rect(x, y, 64, 128));
				string imagepath = savepath + filename + "_%03d.png";
				sprintf(saveName, imagepath.c_str(), j);//生成裁剪出的负样本图片的文件名
				imwrite(saveName, imgROI);//保存文件

				
				Mat imgflip;//反转
				flip(imgROI, imgflip,1);
				sprintf(saveName, (savepath+filename + "_%03d.png").c_str(), j+10);//生成裁剪出的负样本图片的文件名
				imwrite(saveName, imgflip);//保存文件
			}
		}
	}

	system("pause");
}