#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "GainRobustTracker.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	VideoCapture cap(0, cv::CAP_DSHOW);
	Mat frame;
	cv::Mat gray, prevgray;
	GainRobustTracker tracker(11, 3);
	
	std::vector<cv::Point2f> prev_pts, curr_pts, tmp_pts, tmp_pts1;
	std::vector<uchar> status;

	while (waitKey(1) != 27)
	{
		cap >> frame;
		if (frame.empty())
		{
			cap.set(CAP_PROP_POS_FRAMES, 0);
			cap >> frame;
		}
		cv::pyrDown(frame, frame);
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		if(prevgray.empty()) gray.copyTo(prevgray);
		if (prev_pts.size()<50)
		{
			cv::goodFeaturesToTrack(prevgray, prev_pts, 500, 0.03, 20);
		}

		auto start = chrono::steady_clock::now();
		tracker.trackImagePyramids(prevgray, gray, prev_pts, curr_pts, status);
		//calcOpticalFlowPyrLK(prevgray, gray, prev_pts, curr_pts, status, cv::noArray(), Size(11, 11));
		auto end = chrono::steady_clock::now();

		cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << '\n';
		for (int i = 0; i < status.size(); i++)
		{
			if (status[i])
			{
				tmp_pts.push_back(curr_pts[i]);
				tmp_pts1.push_back(prev_pts[i]);
			}
		}
		
		for (int i = 0; i < tmp_pts.size(); i++)
			cv::line(frame, tmp_pts1[i], tmp_pts[i], Scalar(0, status[i] * 255, 255), 5);

		cv::imshow("Optical flow", frame);

		std::swap(prev_pts, tmp_pts);
		tmp_pts.clear();
		tmp_pts1.clear();
		gray.copyTo(prevgray);
	}
	return 0;
}