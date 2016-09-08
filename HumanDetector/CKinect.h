#ifndef CKINECT_H
#define CKINECT_H
#include <iostream>
#include <Kinect.h>
using namespace std;
typedef unsigned short ushort;

class CKinect
{
public:
	bool Init();
	bool UpdateDepth(ushort* dest);
	void Release();
	int getDepthWidth() const
	{
		return width;
	}
	int getDepthHeight() const
	{
		return height;
	}
	~CKinect() { Release(); }
private:
	IKinectSensor* kinectsensor;
	IDepthFrameReader* depthreader;
	IDepthFrameSource* depthsource;
	const int width = 512;
	const int height = 424;
};
#endif
