#pragma once
#ifndef OPENMAKAENGINE_DRAWER_CPP
#define OPENMAKAENGINE_DRAWER_CPP

#include "Drawer.h"
#include "../Controller.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace om;

Drawer::Drawer() {
    if (Controller::MODE_DEBUG) {
        cout << "Creating Drawer instance.." << endl;
    }
}

Drawer::~Drawer() {
    if (Controller::MODE_DEBUG) {
        cout << "Deleting Drawer instance.." << endl;
    }
}

void Drawer::drawContour(Mat& image, vector<Point2f> points2d, Scalar color, int thickness, int lineType, int shift)
{
	// for all points
	for (size_t i = 0; i < points2d.size(); i++)
	{
		// rescale point a coordinates
		Point2f a;
		a.x = static_cast<int>(points2d[i].x + 0.5);
		a.y = static_cast<int>(points2d[i].y + 0.5);

		// rescale point b coordinates
		Point2f b;
		b.x = static_cast<int>(points2d[(i + 1) % points2d.size()].x + 0.5);
		b.y = static_cast<int>(points2d[(i + 1) % points2d.size()].y + 0.5);

		// draw line
		line(image, a, b, color, thickness, lineType, shift);
	}
}

void Drawer::drawContourWithRescale(Mat &image, vector<Point2f> points2d, Scalar color, int thickness,
                         int lineType, int shift) {

    // for all points
    for (size_t i = 0; i < points2d.size(); i++) {

        // rescale point a coordinates
        Point2f a;
        a.x = static_cast<int>(points2d[i].x * SceneFrame::IMAGE_SCALE + 0.5);
        a.y = static_cast<int>(points2d[i].y * SceneFrame::IMAGE_SCALE + 0.5);

        // resale point b coordinates
        Point2f b;
        b.x = static_cast<int>(points2d[(i + 1) % points2d.size()].x * SceneFrame::IMAGE_SCALE + 0.5);
        b.y = static_cast<int>(points2d[(i + 1) % points2d.size()].y * SceneFrame::IMAGE_SCALE + 0.5);

        // draw line
        line(image, a, b, color, thickness, lineType, shift);
    }
}

void Drawer::drawKeypoints(Mat &image, vector<KeyPoint> keyPoints, Scalar color) {

    // for all keypoints
    for (unsigned int i = 0; i < keyPoints.size(); i++) {

        // rescale coordinates
        int x = static_cast<int>((keyPoints[i].pt.x * SceneFrame::IMAGE_SCALE) + 0.5);
        int y = static_cast<int>((keyPoints[i].pt.y * SceneFrame::IMAGE_SCALE) + 0.5);

        // draw circles
        circle(image, Point(x, y), 10, Scalar(255, 0, 0, 255));
    }
}

void Drawer::drawKeypointsWithResponse(Mat &image, vector<KeyPoint> keyPoints, Scalar color) {

    // for all keypoints
    for (unsigned int i = 0; i < keyPoints.size(); i++) {

        // rescale coordinates
	    auto x = static_cast<int>((keyPoints[i].pt.x * SceneFrame::IMAGE_SCALE) + 0.5);
	    auto y = static_cast<int>((keyPoints[i].pt.y * SceneFrame::IMAGE_SCALE) + 0.5);

        // draw circles
        circle(image, Point(x, y), static_cast<int>(keyPoints[i].response + 0.5), Scalar(255, 0, 0, 255));
    }
}

Mat Drawer::drawMatchesWindow(Mat query, Mat pattern, const vector<KeyPoint> &queryKp,
                                  const vector<KeyPoint> &trainKp, vector<DMatch> matches,
                                  int maxMatchesDrawn) {
    Mat outImg;

    if (matches.size() > maxMatchesDrawn) {
        matches.resize(maxMatchesDrawn);
    }

    drawMatches
            (
                    query,
                    queryKp,
                    pattern,
                    trainKp,
                    matches,
                    outImg,
                    Scalar(0, 200, 0, 255),
                    Scalar::all(-1),
                    vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
            );

    return outImg;
}

Mat Drawer::drawKeypointsWindow(Mat query, Mat pattern, const vector<KeyPoint> &queryKp,
                                    const vector<KeyPoint> &trainKp, vector<DMatch> matches,
                                    int maxMatchesDrawn) {
    Mat outImg;

    if (matches.size() > maxMatchesDrawn) {
        matches.resize(maxMatchesDrawn);
    }

    drawMatches
            (
                    query,
                    queryKp,
                    pattern,
                    trainKp,
                    matches,
                    outImg,
                    Scalar(0, 200, 0, 255),
                    Scalar::all(-1),
                    vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
            );

    return outImg;
}

#endif
