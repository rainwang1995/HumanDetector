#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <fstream>
#include "opencvHeader.h"
#include "Utils.h"
#include <cstdlib>
#include <ctime>
#include <string>
using namespace std;
using namespace cv;

int CropImageCount = 0; //裁剪出来的负样本图片个数

int main(int argc, char* argv[])
{
	string negpath = "F:\\liuhao\\DepthDataSet\\Negfull\\";
	string savepath = "F:\\liuhao\\DepthDataSet\\Neg\\";
	vector<string> files;
	if (argc==3)
	{
		negpath = argv[1];
		savepath = argv[2];
	}
	Utils::findallfiles(negpath, files, "png");

	////把yml转换成png
	//char saveName[256];
	//for (int i = 0; i < files.size();++i)
	//{
	//	string ImgName = negpath + files[i];
	//	cout << files[i] << endl;
	//	FileStorage fr(ImgName,FileStorage::READ);
	//	//fr.open(ImgName, FileStorage::READ);
	//	if (fr.isOpened())
	//	{
	//		FileNode Node = fr.getFirstTopLevelNode();
	//		FileNode dataNode = Node["data"];
	//		vector<ushort> data;

	//		if (dataNode.isSeq())
	//		{
	//			//FileNodeIterator it = dataNode.begin(), it_end = dataNode.end();
	//			//for (; it != it_end;++it)
	//			//{
	//			//	//cout << (float)*it;
	//			//	data.push_back((float)*it);
	//			//}
	//			dataNode >> data;
	//		}
	//		//Mat img=Mat_<float>(480, 640);
	//		Mat img(480, 640, CV_16UC1,&data[0]);

	//		//Mat img16;
	//		//img.convertTo(img16, CV_16UC1,&data[0]);

	//		sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//生成裁剪出的负样本图片的文件名
	//		imwrite(saveName, img);//保存文
	//		//cvReleaseImage(&img);
	//	}	
	//}

	Mat src;
	string ImgName;

	char saveName[256];//裁剪出来的负样本图片文件名
	for (int i = 0; i < files.size();++i)
	{
		cout << "处理：" << files[i] << endl;
		ImgName = negpath + files[i];

		src = imread(ImgName, IMREAD_ANYDEPTH);//读取图片
								 //src =cvLoadImage(imagename,1);
								 //cout<<"宽："<<src.cols<<"，高："<<src.rows<<endl;

								 //图片大小应该能能至少包含一个64*128的窗口
		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//设置随机数种子

							  //从每张图片中随机裁剪10个64*128大小的不包含人的负样本
			for (int i = 0; i < 10; i++)
			{
				int x = (rand() % (src.cols - 64)); //x坐标
				int y = (rand() % (src.rows - 128)); //y坐标
													 //cout<<x<<","<<y<<endl;
				Mat imgROI = src(Rect(x, y, 64, 128));
				sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//生成裁剪出的负样本图片的文件名
				imwrite(saveName, imgROI);//保存文件

				Mat imgflip;
				flip(imgROI, imgflip,1);
				sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//生成裁剪出的负样本图片的文件名
				imwrite(saveName, imgROI);//保存文件
			}
		}
	}

	system("pause");
}