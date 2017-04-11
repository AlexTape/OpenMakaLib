#pragma once
#include <opencv2/core/core.hpp>

class Geometry
{
public:

	enum RECTANGLE_MODE
	{
		MAX,
		MIN,
		FIT
	};

	static bool isRectangle(vector<Point2f>& rectanglePoints);
	static vector<Point_<float>> rescale(vector<Point_<float>>& points2d);
	static CvRect fitRectangle(vector<Point_<float>> rectangle, RECTANGLE_MODE mode = MAX);
	static Mat cutRoi(Mat& image, CvRect rect);
	static Mat getRoi(Mat& image, CvRect rect);

private:
	Geometry();
	virtual ~Geometry();
};
