//
// Created by thinker on 27.07.15.
//

#ifndef OPENMAKAENGINE_OBJECTPATTERN_H
#define OPENMAKAENGINE_OBJECTPATTERN_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

namespace om {


    class ObjectPattern {

    public:

	    explicit ObjectPattern(const Mat &grayImage);

        Size size;
        Mat image;

        vector<KeyPoint> keypoints;
        Mat descriptors;

        vector<Point2f> points2d;
        vector<Point3f> points3d;

        virtual ~ObjectPattern(void);

        void build();

    };

}

#endif
