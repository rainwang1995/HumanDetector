#include "CKinect.h"

bool CKinect::Init()
{
	if (FAILED(GetDefaultKinectSensor(&kinectsensor)))
	{
		return false;
	}
	if (kinectsensor)
	{
		if (FAILED(kinectsensor->Open()))
		{
			return false;
		}
		if (FAILED(kinectsensor->get_DepthFrameSource(&depthsource)))
		{
			return false;
		}
		if (FAILED(depthsource->OpenReader(&depthreader)))
		{
			return false;
		}
		else
			return depthreader;
	}
	else
		return false;
}

bool CKinect::UpdateDepth(ushort * dest)
{
	if (depthreader == NULL)
		return false;
	IDepthFrame* depthframe;
	if (FAILED(depthreader->AcquireLatestFrame(&depthframe)))
	{
		return false;
	}
	if (!depthframe)
	{
		return false;
	}
	else
	{
		depthframe->CopyFrameDataToArray(width*height, dest);
		depthframe->Release();
	}
	
	return true;
}

void CKinect::Release()
{
	if (kinectsensor!=NULL)
	{
		kinectsensor->Close();
		kinectsensor->Release();
		kinectsensor = NULL;
	}
	if (depthreader!=NULL)
	{
		depthreader->Release();
		depthreader = NULL;
	}
	if (depthsource!=NULL)
	{
		depthsource->Release();
		depthsource = NULL;
	}
}
