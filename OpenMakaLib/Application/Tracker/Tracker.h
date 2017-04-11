#ifndef OPENMAKAENGINE_TRACKER_H
#define OPENMAKAENGINE_TRACKER_H

//! Big Thanks to Takuya Minagawa
//! source @https://github.com/takmin/OpenCV-Marker-less-AR
//! forked @https://github.com/AlexTape/OpenCV-Marker-less-AR

#include <opencv2/core/core.hpp>

using namespace cv;

namespace om
{
	class Tracker
	{
	public:

		virtual ~Tracker();

		Mat lastImage;
		vector<Point2f> corners;
		vector<Point2f> objectPosition;
		vector<unsigned char> vstatus;

		static int MAX_CORNERS;
		static double QUALITY_LEVEL;
		static double MINIMUM_DISTANCE;

		Mat homography;

		static Mat createAreaMask(Size imageSize, vector<Point2f>& points);

		static vector<Point2f> calcAffineTransformation(vector<Point2f>& pointVector,
		                                                Mat& transformation);

		static bool isObjectInsideImage(Size imageSize, vector<Point2f>& points);

		static bool isRectangle(vector<Point2f>& rectanglePoints);

		static int isInsideArea(vector<Point2f>& points, vector<Point2f>& cornerPoints,
		                        vector<unsigned char>& status);

		static Mat getLastDirection(vector<Point2f>& pointVector);

		void initialize(const Mat& frame,
		                vector<Point2f>& actualObjectPosition);

		bool process(const Mat& sceneImage);

		static Tracker* getInstance();

	private:

		Tracker();

		static Tracker* inst_;
	};
}

#endif
