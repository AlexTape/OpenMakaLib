#ifndef OPENMAKAENGINE_DRAWER_H
#define OPENMAKAENGINE_DRAWER_H

#include <opencv2/core/core.hpp>
#include "../Recognition/ObjectPattern.h"

namespace om
{
	class Drawer
	{
	public:

		static void drawContour(Mat& image, vector<Point2f> points2d, Scalar color, int thickness = 4,
		                        int lineType = 8, int shift = 0);

		static void drawContourWithRescale(Mat& image, vector<Point2f> points2d, Scalar color, int thickness = 4,
		                                   int lineType = 8, int shift = 0);

		static void drawKeypoints(Mat& image, vector<KeyPoint> keyPoints, Scalar color);

		static Mat drawMatchesWindow(Mat query, Mat pattern, const vector<KeyPoint>& queryKp,
		                             const vector<KeyPoint>& trainKp, vector<DMatch> matches,
		                             int maxMatchesDrawn);

		static Mat drawKeypointsWindow(Mat query, Mat pattern, const vector<KeyPoint>& queryKp,
		                               const vector<KeyPoint>& trainKp, vector<DMatch> matches,
		                               int maxMatchesDrawn);

		static void drawKeypointsWithResponse(Mat& image, vector<KeyPoint> keyPoints, Scalar color);

	private:

		Drawer();

		virtual ~Drawer();
	};
}


#endif
