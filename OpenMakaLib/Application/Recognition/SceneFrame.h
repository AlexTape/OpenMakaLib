#ifndef OPENMAKAENGINE_SCENEFRAME_H
#define OPENMAKAENGINE_SCENEFRAME_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

namespace om {

    class SceneFrame {

    public:

        SceneFrame(Mat &rgbInputFrame, Mat &grayInputFrame);

        virtual ~SceneFrame();

        Mat rgb;
        Mat gray;
        Mat homography;

        vector<KeyPoint> keypoints;
        Mat descriptors;
        vector<DMatch> matches;

        vector<Point2f> objectPosition;

        static int MAX_IMAGE_SIZE;
        static float IMAGE_SCALE;

        string getProcessingResolution() const;

        string getInputResolution() const;
    };

}

#endif
