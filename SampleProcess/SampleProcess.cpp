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

int CropImageCount = 0; //�ü������ĸ�����ͼƬ����

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

	////��ymlת����png
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

	//		sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���
	//		imwrite(saveName, img);//������
	//		//cvReleaseImage(&img);
	//	}	
	//}

	Mat src;
	string ImgName;

	char saveName[256];//�ü������ĸ�����ͼƬ�ļ���
	for (int i = 0; i < files.size();++i)
	{
		cout << "����" << files[i] << endl;
		ImgName = negpath + files[i];

		src = imread(ImgName, IMREAD_ANYDEPTH);//��ȡͼƬ
								 //src =cvLoadImage(imagename,1);
								 //cout<<"��"<<src.cols<<"���ߣ�"<<src.rows<<endl;

								 //ͼƬ��СӦ���������ٰ���һ��64*128�Ĵ���
		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//�������������

							  //��ÿ��ͼƬ������ü�10��64*128��С�Ĳ������˵ĸ�����
			for (int i = 0; i < 10; i++)
			{
				int x = (rand() % (src.cols - 64)); //x����
				int y = (rand() % (src.rows - 128)); //y����
													 //cout<<x<<","<<y<<endl;
				Mat imgROI = src(Rect(x, y, 64, 128));
				sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���
				imwrite(saveName, imgROI);//�����ļ�

				Mat imgflip;
				flip(imgROI, imgflip,1);
				sprintf(saveName, (savepath + "neg%06d.png").c_str(), ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���
				imwrite(saveName, imgROI);//�����ļ�
			}
		}
	}

	system("pause");
}