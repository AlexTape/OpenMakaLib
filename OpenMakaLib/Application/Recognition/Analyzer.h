#ifndef OPENMAKAENGINE_ANALYZER_H
#define OPENMAKAENGINE_ANALYZER_H

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "ObjectPattern.h"
#include "SceneFrame.h"
#include "../Helper/Timer.h"

#define AnalyzerTAG "OpenMaka::Analyzer"

namespace om {

    class Analyzer {

    public:

        static string DETECTOR;
        static string EXTRACTOR;
        static string MATCHER;
        static int MINIMUM_INLIERS;
        static int MINIMUM_MATCHES;
        static float NN_DISTANCE_RATIO;
        static double RANSAC_REPROJECTION_THRESHOLD;

        bool IS_BRUTEFORCE_MATCHER;
        static int K_GROUPS;

        virtual ~Analyzer();

        bool isInitialized;

        static Analyzer *getInstance();

        bool releaseOpenCV();

        void initExtractor(string &type);

        void initDetector(string &type);

        void initMatcher(string &type);

        bool initialize();

        bool analyze(Mat &gray, vector<KeyPoint> &keypoints, Mat &descriptors);


        bool process(SceneFrame &sceneFrame);

        bool createObjectPattern(Mat &image);

        bool analyzeSceneFrame(SceneFrame &sceneFrame);

        void matchBinaryDescriptors(SceneFrame &sceneFrame, vector<Point2f> &goodTrainKeypoints,
                                    vector<Point2f> &goodSceneKeypoints);

        void matchFloatDescriptors(SceneFrame &sceneFrame, vector<Point2f> &goodTrainKeypoints,
                                   vector<Point2f> &goodSceneKeypoints);

        bool refineMatches(SceneFrame &query, ObjectPattern &pattern) const;

        void train(const Mat &descriptors);

        void match(SceneFrame &sceneFrame);

        int calcInliers(SceneFrame &sceneFrame, vector<Point2f> &goodTrainKeypoints,
                        vector<Point2f> &goodSceneKeypoints) const;

        bool missingObjectPattern();
		
    private:

        static Analyzer *inst_;

        Analyzer();

        Ptr<FeatureDetector> detector;

        Ptr<DescriptorExtractor> extractor;

        Ptr<DescriptorMatcher> matcher;

        int distance;

        ObjectPattern *activeObjectPattern;

        Timer *clock;

        Timer *timer;


    };

}

#endif
