#pragma once
#ifndef OPENMAKAENGINE_ANALYZER_CPP
#define OPENMAKAENGINE_ANALYZER_CPP

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <iomanip>

#include "Analyzer.h"
#include "../Controller.h"
#include "../Helper/Drawer.h"
#include "../akaze/akaze_features.h"
#include "../Helper/Geometry.h"
#include "../Helper/Int2SizeType.h"

#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define fpu_error(x) (isinf(x) || isnan(x))

using namespace std;
using namespace cv;
using namespace om;


// init static members
Analyzer *Analyzer::inst_ = nullptr;
string Analyzer::DETECTOR;
string Analyzer::EXTRACTOR;
string Analyzer::MATCHER;
int Analyzer::MINIMUM_INLIERS;
int Analyzer::MINIMUM_MATCHES;
float Analyzer::NN_DISTANCE_RATIO;
int Analyzer::K_GROUPS;
double Analyzer::RANSAC_REPROJECTION_THRESHOLD;

Analyzer::Analyzer(): distance(0), IS_BRUTEFORCE_MATCHER(false)
{
	if (Controller::MODE_DEBUG)
	{
		cout << "Creating Analyzer instance.." << endl;
	}

	// init nonfree module for SURF support
	initModule_nonfree();

	// setup measurements
	clock = new Timer();
	timer = new Timer();

	// preinit vars to avoid segmentation faults
	activeObjectPattern = nullptr;

	// set variables
	isInitialized = false;
}

Analyzer::~Analyzer() {
    if (Controller::MODE_DEBUG) {
        cout << "Deleting Analyzer instance.." << endl;
    }
    releaseOpenCV();
    delete activeObjectPattern;
    delete clock;
    delete timer;
}

Analyzer *Analyzer::getInstance() {
    if (inst_ == nullptr) {

        // create singleton instance
        inst_ = new Analyzer();

        // init attributes
        inst_->isInitialized = false;

    }
    return inst_;
}

void Analyzer::initDetector(string &type) {
    if (type == "SIFT") {
        detector = Ptr<FeatureDetector>(new SIFT(400, 3, 0.04, 25, 1.6));
    }
    else if (type == "FAST") {
        detector = Ptr<FeatureDetector>(new FastFeatureDetector(20, true));
    }
    else if (type == "GFTT") {
        detector = Ptr<FeatureDetector>(new GFTTDetector(1000, 0.01, 1, 3, false, 0.04));
    }
    else if (type == "MSER") {
        detector = Ptr<FeatureDetector>(new MSER(5, 60, 14400, 0.25, .2, 200, 1.01, 0.003, 5));
    }
    else if (type == "DENSE") {
        detector = Ptr<FeatureDetector>(new DenseFeatureDetector(1.f, 1, 0.1f, 6, 0, true, false));
    }
    else if (type == "STAR") {
        detector = Ptr<FeatureDetector>(new StarFeatureDetector(45, 30, 10, 8, 5));
    }
    else if (type == "SURF") {
        detector = Ptr<FeatureDetector>(new cv::SURF(600.0, 4, 2, true, false));
    }
    else if (type == "BRISK") {
        detector = Ptr<FeatureDetector>(new BRISK(30, 3, 1.0f));
    }
    else if (type == "ORB") {
        detector = Ptr<FeatureDetector>(new ORB(500, 1.2f, 8, 31,
                                                            0, 2, ORB::HARRIS_SCORE, 31));
    }
    else if (type == "AKAZE") {
        detector = Ptr<FeatureDetector>(new AKAZE());
    }
}

void Analyzer::initExtractor(string &type) {
    if (type == "SIFT") {
        extractor = Ptr<DescriptorExtractor>(new SIFT(0, 3, 0.04, 10, 1.6));
        distance = NORM_L2SQR;
    }
    else if (type == "BRIEF") {
        extractor = Ptr<DescriptorExtractor>(new BriefDescriptorExtractor(32));
        distance = NORM_HAMMING;
    }
    else if (type == "ORB") {
        extractor = Ptr<DescriptorExtractor>(new ORB(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31));
        distance = NORM_HAMMING;
    }
    else if (type == "SURF") {
        extractor = Ptr<DescriptorExtractor>(new cv::SURF(600.0, 4, 2, true, false));
        distance = NORM_L2SQR;
    }
    else if (type == "BRISK") {
        extractor = Ptr<DescriptorExtractor>(new BRISK(30, 3, 1.0f));
        distance = NORM_HAMMING;
    }
    else if (type == "FREAK") {
        extractor = Ptr<DescriptorExtractor>(new FREAK(false, false, 22.0f, 4, vector<int>()));
        distance = NORM_HAMMING;
    }
    else if (type == "AKAZE") {
        extractor = Ptr<DescriptorExtractor>(new AKAZE());
        distance = NORM_HAMMING;
    }
}

void Analyzer::initMatcher(string &type) {

    if (type == "BF") {

        // NOTE: OpenCV Error: Assertion failed (K == 1 && update == 0 && mask.empty()) in batchDistance
        // https://github.com/MasteringOpenCV/code/issues/
        matcher = Ptr<DescriptorMatcher>(new BFMatcher(distance, false));
        // matcher = Ptr<DescriptorMatcher>(new BFMatcher(NORM_HAMMING, false));

        // using bruteforce matching
        IS_BRUTEFORCE_MATCHER = true;

    } else {

        // use flannbased matching
        IS_BRUTEFORCE_MATCHER = false;

    }

	// TODO not implemented yet
    // else if (type == "FLANN_LSF") {
    //  indexParams = new flann::LshIndexParams(12, 20, 2);
    //  searchParams = new flann::SearchParams();
    //  matcher = Ptr<DescriptorMatcher>(new FlannBasedMatcher(indexParams, searchParams));
    //  IS_BRUTEFORCE_MATCHER = false;
    // }
    // else if (type == "FLANN_KD") {
    //  indexParams = new flann::KDTreeIndexParams();
    //  searchParams = new flann::SearchParams();
    //  matcher = Ptr<DescriptorMatcher>(new FlannBasedMatcher(indexParams, searchParams));
    //  IS_BRUTEFORCE_MATCHER = false;
    // }
}

bool Analyzer::analyze(Mat &gray, vector<KeyPoint> &keypoints, Mat &descriptors) {

    bool returnThis = true;

    try {

        if (detector->name() == "Feature2D.SIFT" && extractor->name() == "Feature2D.ORB") {
            // TODO fix testConfigurations.push_back(vector<string>{"SIFT", "ORB", ...});
            // see http://code.opencv.org/issues/2987 and http://code.opencv.org/issues/1277

            // OpenCV Error: Assertion failed (dsize.area() || (inv_scale_x > 0 && inv_scale_y > 0)) in resize, file ../opencv-2.4.11/modules/imgproc/src/imgwarp.cpp, line 1969
            // ../opencv-2.4.11/modules/imgproc/src/imgwarp.cpp:1969: error: (-215) dsize.area() || (inv_scale_x > 0 && inv_scale_y > 0) in function resize

            // is sift-orb supposed to work? Skip this configuration till functionality is fixed..

            // fix test results (dirty way);
            if (IS_BRUTEFORCE_MATCHER) {
                Controller::statistics("Matcher", static_cast<string>("BF"));
            } else {
                Controller::statistics("Matcher", static_cast<string>("FLANN"));
            }
            return false;
        }

        // if there is no image data
        if (Controller::MODE_DEBUG) {
            assert(!gray.empty());
        }
        if (gray.empty()) {
            cvError(0, "Analyzer", "Input image empty!", __FILE__, __LINE__);
        }

        // detect keypoints
        keypoints.clear();
        if (Controller::MODE_DEBUG) {
            assert(keypoints.empty());
        }
        detector->detect(gray, keypoints);

        // no keypoints detected!
        if (keypoints.empty()) {
            // Keep in mind that maybe no features are present in actual image!
            if (Controller::MODE_DEBUG) {
                cvError(0, "Analyzer", "Detection keypoints empty!", __FILE__, __LINE__);
            }
            // no keypoints to compute
            return false;
        }

        // compute descriptors
        descriptors.release();
        if (Controller::MODE_DEBUG) {
            assert(descriptors.empty());
        }

        extractor->compute(gray, keypoints, descriptors);

        // note: keypoints for which a descriptor cannot be computed are removed
        if (keypoints.empty()) {
            if (Controller::MODE_DEBUG) {
                cvError(0, "Analyzer", "Compute keypoints empty!", __FILE__, __LINE__);
            }
            // if all keypoints are removed, no descriptors could be computed
            return false;
        }

    } catch (Exception &exception) {
        if (Controller::MODE_DEBUG) {
            // NOTE: e.g. dark images can contain 0 features!
            cout << exception.what() << endl;
        }
        returnThis = false;
    }

    return returnThis;
}

bool Analyzer::analyzeSceneFrame(SceneFrame &sceneFrame) {

    // validate input data
    if (sceneFrame.gray.empty()) {
        if (Controller::MODE_DEBUG) {
            cvError(0, "Analyzer::process", "Scene image empty!", __FILE__, __LINE__);
        }
        return false;
    }
    if (activeObjectPattern->image.empty()) {
        if (Controller::MODE_DEBUG) {
            cvError(0, "Analyzer::process", "Object image empty!", __FILE__, __LINE__);
        }
        return false;
    }

    analyze(sceneFrame.gray, sceneFrame.keypoints, sceneFrame.descriptors);

    if (Controller::MODE_STATISTICS) {
        Controller::statistics("SceneKeypoints", static_cast<int>(sceneFrame.keypoints.size()));
        Controller::statistics("ObjectKeypoints", static_cast<int>(activeObjectPattern->keypoints.size()));
    }

    if (Controller::MODE_DEBUG) {
        cout << "Keypoints(Scene/Object)=" << sceneFrame.keypoints.size() << "/" <<
        activeObjectPattern->keypoints.size() <<
        endl;
    }

    if (activeObjectPattern->descriptors.empty()) {
        if (Controller::MODE_DEBUG) {
            cout << "object descriptors empty" << endl;
        }
        return false;
    }

    if (sceneFrame.descriptors.empty()) {
        if (Controller::MODE_DEBUG) {
            cout << "scene descriptors empty" << endl;
        }
        return false;
    }

	return true;
}

void Analyzer::matchBinaryDescriptors(SceneFrame &sceneFrame, vector<Point2f> &goodTrainKeypoints,
                                      vector<Point2f> &goodSceneKeypoints) {

    if (Controller::MODE_DEBUG) {
        cout << "Binary descriptors detected..." << endl;
    }

    if (Controller::MODE_STATISTICS) {
        Controller::statistics("DescriptorType", static_cast<string>("binary"));
    }

    timer->restart();

    Mat resultIndex;
    Mat distanceIndex;
    vector<vector<DMatch> > matches;
    vector<int> objectKeypointIndex, sceneKeypointIndex;

    if (IS_BRUTEFORCE_MATCHER) {

        if (Controller::MODE_DEBUG) {
            cout << "BruteForce matching.." << endl;
        }

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("Matcher", static_cast<string>("BF"));
        }

        // TODO train first
        // matcher.radiusMatch(activeObjectPattern->descriptors, sceneFrame.descriptors, matches, 1.0 );
        matcher->knnMatch(activeObjectPattern->descriptors, sceneFrame.descriptors, matches, K_GROUPS);

        for (unsigned int i = 0; i < matches.size(); ++i) {

            if (Controller::MODE_DEBUG) {
                //cout << "DistanceIndex=" << i << "; distance1=" << matches.at(i).at(0).queryIdx << "; distance2=" <<
                //matches.at(i).at(0).distance << ";" << endl;
            }

            unsigned int queryId = abs(matches.at(i).at(0).queryIdx);
            goodTrainKeypoints.push_back(activeObjectPattern->keypoints.at(queryId).pt);
            objectKeypointIndex.push_back(matches.at(i).at(0).queryIdx);

            unsigned int trainId = abs(matches.at(i).at(0).trainIdx);
            goodSceneKeypoints.push_back(sceneFrame.keypoints.at(trainId).pt);
            sceneKeypointIndex.push_back(matches.at(i).at(0).trainIdx);

        }

    } else {

        // Create Flann LSH index
        flann::Index flannIndex(sceneFrame.descriptors, flann::LshIndexParams(12, 20, 2),
                                    cvflann::FLANN_DIST_HAMMING);

        if (Controller::MODE_DEBUG) {
            cout << "Flann LSH Index created.." << endl;
        }

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("Matcher", static_cast<string>("Flann"));
        }

        resultIndex = Mat(activeObjectPattern->descriptors.rows, K_GROUPS, CV_32SC1);
        distanceIndex = Mat(activeObjectPattern->descriptors.rows, K_GROUPS,
                                CV_32FC1);

        flannIndex.knnSearch(activeObjectPattern->descriptors, resultIndex, distanceIndex, K_GROUPS,
                             flann::SearchParams());

        for (int i = 0; i < activeObjectPattern->descriptors.rows; ++i) {

            if (Controller::MODE_DEBUG) {
                //cout << "DistanceIndex=" << i << "; distance1=" << distanceIndex.at<float>(i, 0) << "; distance2=" <<
                //distanceIndex.at<float>(i, 1) << ";" << endl;
            }

            // TODO check if needed!
            if (isnan(distanceIndex.at<float>(i, 0)) || isnan(distanceIndex.at<float>(i, 1))) {
                continue;
            }

            goodTrainKeypoints.push_back(activeObjectPattern->keypoints.at(i).pt);
            objectKeypointIndex.push_back(i);

            unsigned int resultIndexId = abs(resultIndex.at<int>(i, 0));
            goodSceneKeypoints.push_back(sceneFrame.keypoints.at(resultIndexId).pt);
            sceneKeypointIndex.push_back(resultIndex.at<int>(i, 0));
        }
    }

    if (Controller::MODE_DEBUG) {
        cout << "matchingDescriptors=" << timer->getMillis() << endl;
    }

    if (Controller::MODE_STATISTICS) {
        Controller::statistics("MatchingDescriptors(ms)", static_cast<double>(timer->getMillis()));
    }
}

void Analyzer::matchFloatDescriptors(SceneFrame &sceneFrame, vector<Point2f> &goodTrainKeypoints,
                                     vector<Point2f> &goodSceneKeypoints) {

    if (Controller::MODE_DEBUG) {
        cout << "Float descriptors detected..." << endl;
    }

    if (Controller::MODE_STATISTICS) {
        Controller::statistics("DescriptorType", static_cast<string>("float"));
    }

    timer->restart();

    // temp result objects
    Mat resultIndex;
    Mat distanceIndex;
    vector<vector<DMatch> > matches;

    // Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
    // Check if this descriptor matches with those of the objects
    vector<int> objectKeypointIndex, sceneKeypointIndex; // Used for homography

    if (IS_BRUTEFORCE_MATCHER) {

        if (Controller::MODE_DEBUG) {
            cout << "BruteForce NORM_L2 matching.." << endl;
        }

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("Matcher", static_cast<string>("BF"));
        }

        // knnMatch
        matcher->knnMatch(activeObjectPattern->descriptors, sceneFrame.descriptors, matches, K_GROUPS);

        try {

            for (unsigned int i = 0; i < matches.size(); ++i) {

                if (Controller::MODE_DEBUG) {
                    //cout << "DistanceIndex=" << i << "; distance1=" << matches.at(i).at(0).queryIdx << "; distance2=" <<
                    //matches.at(i).at(0).distance << ";" << endl;
                }

                if (matches.at(i).at(0).distance <= NN_DISTANCE_RATIO * matches.at(i).at(1).distance) {

                    unsigned int queryId = abs(matches.at(i).at(0).queryIdx);
                    goodTrainKeypoints.push_back(activeObjectPattern->keypoints.at(queryId).pt);
                    objectKeypointIndex.push_back(matches.at(i).at(0).queryIdx);

                    unsigned int trainId = abs(matches.at(i).at(0).trainIdx);
                    goodSceneKeypoints.push_back(sceneFrame.keypoints.at(trainId).pt);
                    sceneKeypointIndex.push_back(matches.at(i).at(0).trainIdx);
                }
            }

        } catch (out_of_range &exception) {
            // fix this error:
            // terminate called after throwing an instance of 'out_of_range'
            // what():  vector::_M_range_check
            if (Controller::MODE_DEBUG) {
                cvError(0, "Analyzer", "Matches out of range!", __FILE__, __LINE__);
                cout << exception.what() << endl;
            }
        }

    } else {

        // Create Flann KDTree index
        flann::Index flannIndex(sceneFrame.descriptors, flann::KDTreeIndexParams(),
                                    cvflann::FLANN_DIST_EUCLIDEAN);

        if (Controller::MODE_DEBUG) {
            cout << "Flann KDTree created.." << endl;
        }

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("Matcher", static_cast<string>("Flann"));
        }

        resultIndex = Mat(activeObjectPattern->descriptors.rows, K_GROUPS, CV_32SC1); // Results index
        distanceIndex = Mat(activeObjectPattern->descriptors.rows, K_GROUPS,
                                CV_32FC1); // Distance resultIndex are CV_32FC1

        // search (nearest neighbor)
        flannIndex.knnSearch(activeObjectPattern->descriptors, resultIndex, distanceIndex, K_GROUPS,
                             flann::SearchParams());

        try {

            for (int i = 0; i < activeObjectPattern->descriptors.rows; ++i) {

                if (Controller::MODE_DEBUG) {
                    //cout << "DistanceIndex=" << i << "; distance1=" << distanceIndex.at<float>(i, 0) << "; distance2=" <<
                    //distanceIndex.at<float>(i, 1) << ";" << endl;
                }

                // TODO check if needed!
                if (isnan(distanceIndex.at<float>(i, 0)) || isnan(distanceIndex.at<float>(i, 1))) {
                    continue;
                }

                if (distanceIndex.at<float>(i, 0) <= NN_DISTANCE_RATIO * distanceIndex.at<float>(i, 1)) {
                    goodTrainKeypoints.push_back(activeObjectPattern->keypoints.at(i).pt);
                    objectKeypointIndex.push_back(i);

                    unsigned int resultIndexId = abs(resultIndex.at<int>(i, 0));
                    goodSceneKeypoints.push_back(sceneFrame.keypoints.at(resultIndexId).pt);
                    sceneKeypointIndex.push_back(resultIndex.at<int>(i, 0));
                }
            }

        } catch (out_of_range &exception) {
            // fix this error:
            // terminate called after throwing an instance of 'out_of_range'
            // what():  vector::_M_range_check
            if (Controller::MODE_DEBUG) {
                cvError(0, "Analyzer", "Matches out of range!", __FILE__, __LINE__);
                cout << exception.what() << endl;
            }
        }
    }

    if (Controller::MODE_DEBUG) {
        cout << "matchingDescriptors=" << timer->getMillis() << endl;
    }

    if (Controller::MODE_STATISTICS) {
        Controller::statistics("MatchingDescriptors(ms)", static_cast<double>(timer->getMillis()));
    }

}

bool Analyzer::process(SceneFrame &sceneFrame) {

    // calc processing time for this method
    if (Controller::MODE_STATISTICS) {
        clock->restart();
    }

    bool enoughInliers = false;

    if (isInitialized) {

        if (Controller::MODE_STATISTICS) {
            timer->restart();
        }

        // analyze features and descriptors for frame
        analyzeSceneFrame(sceneFrame);

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("AnalyzeSceneFrame(ms)", static_cast<double>(timer->getMillis()));
        }

        vector<Point2f> goodTrainKeypoints;
        vector<Point2f> goodSceneKeypoints;

        // check preconditions
        // there have to be descriptors! (catch black screens etc)
        if (activeObjectPattern->descriptors.rows <= 0) {
            return false;
        }
        if (sceneFrame.descriptors.rows <= 0) {
            return false;
        }

        if (activeObjectPattern->descriptors.type() == CV_8U) {
            matchBinaryDescriptors(sceneFrame, goodTrainKeypoints, goodSceneKeypoints);
        }

        if (activeObjectPattern->descriptors.type() == CV_32F) {
            matchFloatDescriptors(sceneFrame, goodTrainKeypoints, goodSceneKeypoints);
        }

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("GoodTrainKeypoints", static_cast<int>(goodTrainKeypoints.size()));
            Controller::statistics("GoodSceneKeypoints", static_cast<int>(goodSceneKeypoints.size()));
        }

        if (goodSceneKeypoints.size() != goodTrainKeypoints.size()) {
            if (Controller::MODE_DEBUG) {
                cvError(0, "Analyzer", "goodScene and goodObject keypoints size NOT equal!", __FILE__, __LINE__);
            }
            // not matched!
            return false;
        }

	    auto inliers = 0;

        // NOTE: minimum 4 points needed to calc homography
		Int2SizeType _MINIMUM_MATCHES (MINIMUM_MATCHES);
        if (_MINIMUM_MATCHES <= goodSceneKeypoints.size()) {
            if (goodSceneKeypoints.size() == goodTrainKeypoints.size()) {

                if (Controller::MODE_STATISTICS) {
                    timer->restart();
                }

                // calculate inliers
                inliers = calcInliers(sceneFrame, goodTrainKeypoints, goodSceneKeypoints);

                if (Controller::MODE_STATISTICS) {
                    Controller::statistics("InliersCalc(ms)", static_cast<double>(timer->getMillis()));
                }

            } else {

                if (Controller::MODE_DEBUG) {
                    cout << "GoodSceneKeypoints != GoodTranKeypoints" << endl;
                }

            }
        } else {

            if (Controller::MODE_DEBUG) {
                cout << "Not enough keypoint matches (" << goodTrainKeypoints.size() << "/" << MINIMUM_MATCHES <<
                ") for homography.." << endl;
            }

        }

        enoughInliers = inliers >= MINIMUM_INLIERS;

        // find perspective and draw rectangle
        if (enoughInliers) {

            if (Controller::MODE_STATISTICS) {
                timer->restart();
            }

            // calulate perspective transformation
            perspectiveTransform(activeObjectPattern->points2d, sceneFrame.objectPosition, sceneFrame.homography);

            if (Controller::MODE_STATISTICS) {
                Controller::statistics("PerspectiveTransform(ms)", static_cast<double>(timer->getMillis()));
            }

            if (Controller::MODE_USE_WINDOWS) {

                // drawing contours
                Drawer::drawContourWithRescale(sceneFrame.gray, sceneFrame.objectPosition, Scalar(0, 255, 0));

                //-- Show detected matches
//        Mat Drawer::drawMatchesWindow(Mat query, Mat pattern, const vector<KeyPoint> &queryKp,
//                                           const vector<KeyPoint> &trainKp, vector<DMatch> matches,
//                                           int maxMatchesDrawn) {

                // open custom windows
				string window_name = DETECTOR + "-" + EXTRACTOR + "-" + MATCHER;
				namedWindow(window_name, 0); //resizable window;
	            resizeWindow(window_name,800,800);
				imshow(window_name, sceneFrame.gray);
            }

        } else {

            if (Controller::MODE_DEBUG) {
                cout << "Not enough inliers (" << inliers << "/" << MINIMUM_INLIERS << ") calculating perspective" <<
                endl;
            }

        }

        if (Controller::MODE_STATISTICS) {
            Controller::statistics("AnalyzerProcess(ms)", static_cast<double>(clock->getMillis()));
        }

    } else {
        if (Controller::MODE_DEBUG) {
            cvError(0, "Analyzer", "Analyzer not initialized!", __FILE__, __LINE__);
        }
    }

    // basic rule to devide if object was found or not
    bool objectFound = false;
    if (enoughInliers) {
        objectFound = Geometry::isRectangle(sceneFrame.objectPosition);
    }

    return objectFound;
}



int Analyzer::calcInliers(SceneFrame &sceneFrame, vector<Point2f> &goodTrainKeypoints,
                          vector<Point2f> &goodSceneKeypoints) const
{

    vector<uchar> outlierMask;  // Used for homography

    sceneFrame.homography = findHomography(goodTrainKeypoints,
                                           goodSceneKeypoints,
                                           RANSAC,
                                           RANSAC_REPROJECTION_THRESHOLD,
                                           outlierMask);

    int inliers = 0, outliers = 0;
    for (unsigned int k = 0; k < goodTrainKeypoints.size(); ++k) {
        if (outlierMask.at(k)) {
            ++inliers;
        }
        else {
            ++outliers;
        }
    }

    if (Controller::MODE_DEBUG) {
        cout << "calcInliers: Inliers=" << inliers << "; Outliers=" << outliers << ";" << endl;
    }

    if (Controller::MODE_STATISTICS) {
        Controller::statistics("Inliers", static_cast<int>(inliers));
        Controller::statistics("Outliers", static_cast<int>(outliers));
    }

    return inliers;
}

void Analyzer::match(SceneFrame &query) {
    assert(!query.descriptors.empty());
//    query.matches.clear();
    matcher->match(query.descriptors, query.matches);
}

void Analyzer::train(const Mat &descriptors) {

    // clear old training data
    matcher->clear();

    // TODO register pattern to list and extends descriptor vector to hold multiple patterns?
    // create a descriptor vector to store descriptor data
    vector<Mat> descriptorList(1);
    descriptorList[0] = descriptors.clone();

    // promote descriptor list to matcher instance
    matcher->add(descriptorList);

    // train matcher
    matcher->train();
}

// TODO überarbeiten
bool Analyzer::refineMatches
        (SceneFrame &query, ObjectPattern &pattern) const
{

    const int minNumberMatchesAllowed = 8;

    // TODO make matching working..
    cout << "RefineMatches [START]: SceneFrame Keypoints: " << setw(4) << query.keypoints.size() <<
    "; SceneFrame Matches: " << setw(4) <<
    query.matches.size() << endl;

    vector<KeyPoint> trainKeypoints = pattern.keypoints;

    if (query.matches.size() < minNumberMatchesAllowed)
        return false;

    // Prepare data for findHomography
    vector<Point2f> srcPoints(query.matches.size());
    vector<Point2f> dstPoints(query.matches.size());

    for (size_t i = 0; i < query.matches.size(); i++) {

        cout << pattern.keypoints[i].response << endl;

    }

    for (size_t i = 0; i < query.matches.size(); i++) {

        srcPoints[i] = pattern.keypoints[query.matches[i].trainIdx].pt;
        dstPoints[i] = query.keypoints[query.matches[i].queryIdx].pt;
    }

    cout << srcPoints.size() << "::" << dstPoints.size() << endl;

    // Find homography matrix and get inliers mask
    vector<unsigned char> inliersMask(srcPoints.size());
    Mat homography = findHomography(srcPoints,
                                            dstPoints,
                                            CV_FM_RANSAC,
                                            3.0,
                                            inliersMask);

    // TODO update homo here
    //trackerObject.homography = &homography;

    vector<DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++) {
        if (inliersMask[i]) {
            inliers.push_back(query.matches[i]);
        }
    }

    query.matches.swap(inliers);

    cout << "RefineMatches [ END ]: SceneFrame Keypoints: " << setw(4) << query.keypoints.size() <<
    "; SceneFrame Matches: " << setw(4) <<
    query.matches.size() << endl;

    return query.matches.size() > minNumberMatchesAllowed;

}

bool Analyzer::initialize() {
    if (!isInitialized) {

        // delete activeObjectPattern
        delete activeObjectPattern;
        activeObjectPattern = nullptr;

        // clear detector/extractor instances if needed
        if (detector || extractor) {
            releaseOpenCV();
        }

        // init detector/extractor
        if (!DETECTOR.empty() && !EXTRACTOR.empty() && !MATCHER.empty()) {
            initDetector(DETECTOR);
            initExtractor(EXTRACTOR);
            initMatcher(MATCHER);
        } else {
            return false;
        }

        // done
        isInitialized = true;
    }
    return isInitialized;
}

bool Analyzer::releaseOpenCV() {
    detector.release();
    extractor.release();
    matcher.release();
    return true;
}

bool Analyzer::createObjectPattern(Mat &gray) {

    bool returnThis = false;

    if (!gray.empty()) {
        delete activeObjectPattern;
        activeObjectPattern = new ObjectPattern(gray);
        if (activeObjectPattern && isInitialized) {
            returnThis = analyze(activeObjectPattern->image, activeObjectPattern->keypoints,
                                 activeObjectPattern->descriptors);
        }
    } else {
        if (Controller::MODE_DEBUG) {
            cvError(0, "Analyzer::createObjectPattern", "Image empty!", __FILE__, __LINE__);
        }
    }

    return returnThis;
}

bool Analyzer::missingObjectPattern() {
    bool returnThis = true;
    if (!activeObjectPattern) {
        Mat objectImage = imread(Controller::STORAGE_PATH + Controller::DEFAULT_OBJECT_IMAGE,
                                         CV_LOAD_IMAGE_GRAYSCALE);
        createObjectPattern(objectImage);
    } else {
        returnThis = false;
    }
    return returnThis;
}

#endif
