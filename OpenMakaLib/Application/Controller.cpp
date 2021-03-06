#pragma once

#ifndef OPENMAKAENGINE_CONTROLLER_CPP
#define OPENMAKAENGINE_CONTROLLER_CPP

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iomanip>
#include <iostream>

#include "Controller.h"
#include "Helper/Drawer.h"

#ifdef __ANDROID__
#include "androidbuf.h"
#endif

using namespace std;
using namespace cv;
using namespace om;

// init static members
Controller* Controller::inst_ = nullptr;
bool Controller::MODE_OBJECT_DETECTION;
bool Controller::MODE_TRACKING;
bool Controller::MODE_OPENGL;
bool Controller::MODE_STATISTICS;
bool Controller::MODE_DEBUG;
bool Controller::MODE_USE_WINDOWS;
bool Controller::MODE_SAVE_RESULT_FRAMES;
Analyzer* Controller::analyzer;
Tracker* Controller::tracker;
Statistics* Controller::stats;
bool Controller::isInitialized;
Size Controller::FRAME_SIZE;
string Controller::STORAGE_PATH;
string Controller::CONFIG_FILE;
string Controller::DEFAULT_OBJECT_IMAGE;
string Controller::STATISTICS_FILE;


Controller::Controller()
{
	if (MODE_DEBUG)
	{
		cout << "Creating Controller instance.." << endl;
	}

	// define analyzer instance
	analyzer = Analyzer::getInstance();

	// define tracker instance
	tracker = Tracker::getInstance();

	// initialize variables
	sceneFrame = nullptr;

	// set acutal state
	isInitialized = false;

	// set environment triggers
	MODE_OBJECT_DETECTION = false;
	MODE_TRACKING = false;
	MODE_OPENGL = false;

	// set state variables
	STATE_OBJECT_FOUND = false;
	STATE_TRACKING_OBJECT = false;
	STATE_DISPLAY_OPENGL = false;

	// create statistics instance
	stats = new Statistics();

	// create global clock
	clock = new Timer();

	// create local timer
	timer = new Timer();

#ifdef __ANDROID__
	// make cout working in logcat
    cout.rdbuf(new androidbuf);
#endif
}

Controller::~Controller()
{
	if (MODE_DEBUG)
	{
		cout << "Deleting Controller instance.." << endl;
	}
	delete tracker;
	delete analyzer;
	delete sceneFrame;
	delete stats;
	delete clock;
	delete timer;

#ifdef __ANDROID__
    delete cout.rdbuf(0);
#endif
}

Controller* Controller::getInstance()
{
	if (inst_ == nullptr)
	{
		inst_ = new Controller();
	}
	return inst_;
}

// NEEDED
int Controller::initialize(Mat& frame, string storagePath, string configFile) const
{
	// set default config file
	CONFIG_FILE = configFile;

	// initializing..
	isInitialized = false;

	// grab frame size
	FRAME_SIZE = Size(frame.cols, frame.rows);

	// define storage path
	STORAGE_PATH = storagePath;

	// load config file
	FileStorage storage(STORAGE_PATH + CONFIG_FILE, FileStorage::READ);

	// load default image/pattern
	DEFAULT_OBJECT_IMAGE = static_cast<string>(storage["defaultObjectImage"]);

	// load statistics file path
	STATISTICS_FILE = static_cast<string>(storage["statisticsFile"]);

	// load environment variables
	string statistics = static_cast<string>(storage["statisticsMode"]);
	MODE_STATISTICS = statistics == "true";
	string debug = static_cast<string>(storage["debugMode"]);
	MODE_DEBUG = debug == "true";
	string windows = static_cast<string>(storage["useWindows"]);
	MODE_USE_WINDOWS = windows == "true";
	string saveFrames = static_cast<string>(storage["saveResultFrames"]);
	MODE_SAVE_RESULT_FRAMES = saveFrames == "true";

	// load scene frame attributes
	FileNode sceneFrameNode = storage["sceneFrame"];
	SceneFrame::MAX_IMAGE_SIZE = static_cast<int>(sceneFrameNode["maxImageSize"]);

	// load analyzer attributes
	FileNode analyzerNode = storage["analyzer"];
	Analyzer::DETECTOR = static_cast<string>(analyzerNode["detector"]);
	Analyzer::EXTRACTOR = static_cast<string>(analyzerNode["extractor"]);
	Analyzer::MATCHER = static_cast<string>(analyzerNode["matcher"]);
	Analyzer::MINIMUM_INLIERS = static_cast<int>(analyzerNode["minimumInliers"]);
	Analyzer::MINIMUM_MATCHES = static_cast<int>(analyzerNode["minimumMatches"]);
	Analyzer::NN_DISTANCE_RATIO = static_cast<float>(analyzerNode["nnDistanceRatio"]);
	Analyzer::K_GROUPS = static_cast<int>(analyzerNode["kGroups"]);
	Analyzer::RANSAC_REPROJECTION_THRESHOLD = static_cast<double>(analyzerNode["ransacReprojectionThreshold"]);

	// initialize analyzer
	analyzer->initialize();

	// load tracker attributes
	FileNode trackerNode = storage["tracker"];
	Tracker::MAX_CORNERS = static_cast<int>(trackerNode["maxCorners"]);
	Tracker::QUALITY_LEVEL = static_cast<double>(trackerNode["qualityLevel"]);
	Tracker::MINIMUM_DISTANCE = static_cast<double>(trackerNode["minimumDistance"]);

	// add default object
	Mat objectImage = imread(STORAGE_PATH + DEFAULT_OBJECT_IMAGE,
	                         CV_LOAD_IMAGE_GRAYSCALE);
	analyzer->createObjectPattern(objectImage);

	// release storage
	if (storage.isOpened())
	{
		storage.release();
	}

	if (MODE_DEBUG)
	{
		cout << "Loading attributes.." << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "C O N T R O L L E R" << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "::STORAGE_PATH=" << Controller::STORAGE_PATH << endl;
		cout << "::FRAME_SIZE=" << Controller::FRAME_SIZE << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "S C E N E   F R A M E" << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "::MAX_IMAGE_SIZE=" << SceneFrame::MAX_IMAGE_SIZE << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "A N A L Y Z E R" << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "::DETECTOR=" << Analyzer::DETECTOR << endl;
		cout << "::EXTRACTOR=" << Analyzer::EXTRACTOR << endl;
		cout << "::MATCHER=" << Analyzer::MATCHER << endl;
		cout << "::MINIMUM_INLIERS=" << Analyzer::MINIMUM_INLIERS << endl;
		cout << "::MINIMUM_MATCHES=" << Analyzer::MINIMUM_MATCHES << endl;
		cout << "::NN_DISTANCE_RATIO=" << Analyzer::NN_DISTANCE_RATIO << endl;
		cout << "::K_GROUPS=" << Analyzer::K_GROUPS << endl;
		cout << "::RANSAC_REPROJECTION_THRESHOLD=" << Analyzer::RANSAC_REPROJECTION_THRESHOLD << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "T R A C K E R" << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "::MAX_CORNERS=" << Tracker::MAX_CORNERS << endl;
		cout << "::QUALITY_LEVEL=" << Tracker::QUALITY_LEVEL << endl;
		cout << "::MINIMUM_DISTANCE=" << Tracker::MINIMUM_DISTANCE << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "Loading done!" << endl;
	}

	// instance is initialized
	isInitialized = true;

	// if ok
	return 1;
}

int Controller::displayFunction(Mat& mRgbaFrame, Mat& mGrayFrame, string imageName)
{
	if (MODE_STATISTICS)
	{
		clock->restart();
		stats->reset();
		statistics("ImageName", static_cast<string>(imageName));
		statistics("Detector", static_cast<string>(Analyzer::DETECTOR));
		statistics("Extractor", static_cast<string>(Analyzer::EXTRACTOR));
	}

	int returnThis = 0;

	// if object detection is enabled..
	if (MODE_OBJECT_DETECTION)
	{
		// recreate object pattern if it is not existing
		if (analyzer->missingObjectPattern())
		{
			if (MODE_DEBUG)
			{
				cout << "ObjectPattern (re-) created!" << endl;
			}
		}

		// ..and tracking disabled
		if (!MODE_TRACKING)
		{
			// delete old scene frame
			delete sceneFrame;

			// create new scene frame
			sceneFrame = new SceneFrame(mRgbaFrame, mGrayFrame);

			if (MODE_STATISTICS)
			{
				statistics("InputResolution",
				           static_cast<string>(sceneFrame->getInputResolution()));
				statistics("ProcessingResolution",
				           static_cast<string>(sceneFrame->getProcessingResolution()));
				timer->restart();
			}

			// analyzer processing
			STATE_OBJECT_FOUND = analyzer->process(*sceneFrame);

			// write statistics for analyzer in non-tracking mode
			if (MODE_STATISTICS)
			{
				statistics("AnalyzerProcess(ms)", static_cast<double>(timer->getMillis()));
				statistics("ObjectFound", static_cast<bool>(STATE_OBJECT_FOUND));
			}

			// drawing green contours
			Drawer::drawContourWithRescale(mRgbaFrame, sceneFrame->objectPosition, Scalar(0, 255, 0));
		}

		// ..and tracking enabled
		if (MODE_TRACKING)
		{
			// if object NOT is trackable
			if (!STATE_TRACKING_OBJECT)
			{
				// delete old scene frame
				delete sceneFrame;

				// create new scene frame
				sceneFrame = new SceneFrame(mRgbaFrame, mGrayFrame);

				if (MODE_STATISTICS)
				{
					statistics("InputResolution",
					           static_cast<string>(sceneFrame->getInputResolution()));
					statistics("ProcessingResolution",
					           static_cast<string>(sceneFrame->getProcessingResolution()));
					timer->restart();
				}

				// analyzer processing
				STATE_OBJECT_FOUND = analyzer->process(*sceneFrame);

				// write statistics for analyzer in tracking mode
				if (MODE_STATISTICS)
				{
					statistics("AnalyzerProcess(ms)", static_cast<double>(timer->getMillis()));
					statistics("ObjectFound", static_cast<bool>(STATE_OBJECT_FOUND));
				}

				// if object found
				if (STATE_OBJECT_FOUND)
				{
					// resize gray image
					Mat trackerFrame = sceneFrame->gray;
					try
					{
						resize(mGrayFrame, trackerFrame, trackerFrame.size());
					}
					catch (Exception& exception)
					{
						cvError(0, "TrackerFrame", "Resizing failed!", __FILE__, __LINE__);
						cout << exception.what() << endl;
					}

					if (MODE_STATISTICS)
					{
						statistics("InputResolution",
						           static_cast<string>(sceneFrame->getInputResolution()));
						statistics("ProcessingResolution",
						           static_cast<string>(sceneFrame->getProcessingResolution()));
						timer->restart();
					}

					// analyzer processing
					bool isInImage = tracker->isObjectInsideImage(trackerFrame.size(), sceneFrame->objectPosition);

					if (MODE_STATISTICS)
					{
						statistics("isObjectInsideImage(ms)", static_cast<double>(timer->getMillis()));
						statistics("isInImage", static_cast<bool>(isInImage));
					}

					if (isInImage)
					{
						if (MODE_STATISTICS)
						{
							timer->restart();
						}

						// initialize tracking
						tracker->initialize(trackerFrame, sceneFrame->objectPosition);

						if (MODE_STATISTICS)
						{
							statistics("TrackerInitialize(ms)", static_cast<double>(timer->getMillis()));
						}

						// can track object
						STATE_TRACKING_OBJECT = true;
					}
					else
					{
						// drawing red contours
						Drawer::drawContourWithRescale(mRgbaFrame, sceneFrame->objectPosition, Scalar(0, 0, 255));

						// could not track object
						STATE_TRACKING_OBJECT = false;
					}
				}
			}

			// if object IS trackable
			if (STATE_TRACKING_OBJECT)
			{
				// resize gray image
				Mat trackerFrame = sceneFrame->gray;
				try
				{
					resize(mGrayFrame, trackerFrame, trackerFrame.size());
				}
				catch (Exception& exception)
				{
					cvError(0, "TrackerFrame", "Resizing failed!", __FILE__, __LINE__);
					cout << exception.what() << endl;
				}

				if (MODE_STATISTICS)
				{
					timer->restart();
				}

				// processing
				STATE_TRACKING_OBJECT = tracker->process(trackerFrame);

				if (MODE_STATISTICS)
				{
					statistics("TrackingProcess(ms)", static_cast<double>(timer->getMillis()));
				}

				// draw blue contours
				Drawer::drawContourWithRescale(mRgbaFrame, tracker->objectPosition, Scalar(255, 0, 0));
			}
		}

		if (MODE_STATISTICS)
		{
			statistics("DisplayFunction(ms)", static_cast<double>(clock->getMillis()));
			stats->write(STATISTICS_FILE);
		}
	}

	// add text to window(s)
	if (MODE_USE_WINDOWS || MODE_SAVE_RESULT_FRAMES)
	{
		// create text
		char text[255];
		string gotObject;
		if (STATE_OBJECT_FOUND)
		{
			gotObject = "true";
		}
		else
		{
			gotObject = "false";
		}
		sprintf(text, "%s-%s-%s Found:%s", Analyzer::DETECTOR.c_str(), Analyzer::EXTRACTOR.c_str(),
		        Analyzer::MATCHER.c_str(), gotObject.c_str());

		// draw text background (white)
		rectangle(mRgbaFrame, Point(0, 0), Point(305, 25), CV_RGB(255, 255, 255), -1);

		// draw text
		putText(mRgbaFrame, text, Point(10, 15), CV_FONT_HERSHEY_PLAIN,
		        1,
		        CV_RGB(255, 0, 0));

		// display image?
		if (MODE_USE_WINDOWS)
		{
			string window_name = Analyzer::DETECTOR + "-" + Analyzer::EXTRACTOR + "-" + Analyzer::MATCHER;
			namedWindow(window_name, 0); //resizable window;
			resizeWindow(window_name, 800, 800);
			imshow(window_name, mRgbaFrame);
		}

		// save image?
		if (MODE_SAVE_RESULT_FRAMES)
		{
			string imagepath = STORAGE_PATH + "\\test-results\\" + imageName + "-" + Analyzer::DETECTOR
				+ "-" + Analyzer::EXTRACTOR + "-" + Analyzer::MATCHER + ".jpg";
			cout << "Write result-image to: " + imagepath << endl;
			imwrite(imagepath, mRgbaFrame);
		}
	}

	// return state
	if (STATE_OBJECT_FOUND || STATE_TRACKING_OBJECT)
	{
		returnThis = 1;
	}

	return returnThis;
}

int Controller::setDetector(string type) const
{
	int returnThis = 0;
	bool result = configure(type, Analyzer::EXTRACTOR, Analyzer::MATCHER);
	if (result)
	{
		returnThis = 1;
	}
	return returnThis;
}

int Controller::setExtractor(string type) const
{
	int returnThis = 0;
	bool result = configure(Analyzer::DETECTOR, type, Analyzer::MATCHER);
	if (result)
	{
		returnThis = 1;
	}
	return returnThis;;
}

int Controller::setMatcher(string type) const
{
	int returnThis = 0;
	bool result = configure(Analyzer::DETECTOR, Analyzer::EXTRACTOR, type);
	if (result)
	{
		returnThis = 1;
	}
	return returnThis;
}

void Controller::isModeObjectDetection(bool isActive)
{
	//    log(ControllerTAG, "MODE_OBJECT_DETECTION: %b", isActive);
	MODE_OBJECT_DETECTION = isActive;
}

void Controller::isModeTracking(bool isActive)
{
	//    log(ControllerTAG, "MODE_TRACKING: %b", isActive);
	MODE_TRACKING = isActive;
}

void Controller::isModeOpenGL(bool isActive)
{
	//    log(ControllerTAG, "MODE_OPENGL: %b", isActive);
	MODE_OPENGL = isActive;
}

void Controller::isModeDebug(bool isActive)
{
	//    log(ControllerTAG, "MODE_OPENGL: %b", isActive);
	MODE_DEBUG = isActive;
}

void Controller::isModeStatistics(bool isActive)
{
	//    log(ControllerTAG, "MODE_OPENGL: %b", isActive);
	MODE_STATISTICS = isActive;
}

bool Controller::createObjectPattern(Mat& rgb, Mat& gray)
{
	// register object pattern to analyzer
	return analyzer->createObjectPattern(gray);
}

bool Controller::configure(string detector, string extractor, string matcher) const
{
	// disable analyzer
	analyzer->isInitialized = false;
	isInitialized = false;

	// set values
	Analyzer::DETECTOR = detector;
	Analyzer::EXTRACTOR = extractor;
	Analyzer::MATCHER = matcher;

	// update analyzer
	bool returnThis = analyzer->initialize();

	if (MODE_DEBUG && returnThis)
	{
		cout << "Controller initialized [Detector=" << Analyzer::DETECTOR << ", Extractor=" << Analyzer::EXTRACTOR <<
			", Matcher" << Analyzer::MATCHER << "]" << endl;
	}

	return returnThis;
}

int Controller::test(vector<string> images, int test, int quantifier)
{
	Mat sceneRgbImageData, sceneGrayImageData, objectRgbImage, objectGrayImage;

	// prepare object image
	cout << "Loading Object " << STORAGE_PATH + images.at(0) << endl;
	objectRgbImage = imread(STORAGE_PATH + images.at(0));
	if (objectRgbImage.empty())
	{
		cout << "Object image cannot be read" << endl;
		return 2;
	}
	cvtColor(objectRgbImage, objectGrayImage, CV_RGB2GRAY);

	// prepare first scene image
	cout << "Loading Scene " << STORAGE_PATH + images.at(1) << endl;
	sceneRgbImageData = imread(STORAGE_PATH + images.at(1));
	if (sceneRgbImageData.empty())
	{
		cout << "Scene image cannot be read" << endl;
		return 1;
	}
	cvtColor(sceneRgbImageData, sceneGrayImageData, CV_RGB2GRAY);

	// initialize
	if (!isInitialized)
	{
		initialize(sceneRgbImageData, STORAGE_PATH, "\\config\\config.xml");
	}

	// set testing mode and save actual configuration
	bool wasObjectDetection = MODE_OBJECT_DETECTION;
	bool wasTracking = MODE_TRACKING;
	bool wasStatistics = MODE_STATISTICS;
	MODE_OBJECT_DETECTION = true;
	MODE_TRACKING = false;
	MODE_STATISTICS = true;

	if (MODE_DEBUG)
	{
		cout << "-----------------------------------------------------" << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << " T E S T I N G   S U I T E" << endl;
		cout << "-----------------------------------------------------" << endl;
	}

	// some variables to control testing routine
	int doRuns = quantifier;

	// default test configuration
	bool doSIFT = false;
	bool doFAST = false;
	bool doGFTT = false;
	bool doMSER = false;
	bool doDENSE = false;
	bool doSTAR = false;
	bool doSURF = false;
	bool doBRISK = false;
	bool doORB = false;
	bool doAKAZE = false;

	// trigger tests
	switch (test)
	{
	case 0:
		doSIFT = true;
		doFAST = true;
		doGFTT = true;
		doMSER = true;
		doDENSE = true;
		doSTAR = true;
		doSURF = true;
		doBRISK = true;
		doORB = true;
		doAKAZE = true;
		break;
	case 1:
		doSIFT = true;
		break;
	case 2:
		doFAST = true;
		break;
	case 3:
		doGFTT = true;
		break;
	case 4:
		doMSER = true;
		break;
	case 5:
		doDENSE = true;
		break;
	case 6:
		doSTAR = true;
		break;
	case 7:
		doSURF = true;
		break;
	case 8:
		doBRISK = true;
		break;
	case 9:
		doORB = true;
		break;
	case 10:
		doAKAZE = true;
		break;
	default:
		return 0;
	}

	// init variables
	Mat sceneRgbImage, sceneGrayImage;
	vector<Configuration> testConfigurations;
	Configuration conf;

	// NOTE: c++10 is not supporting initializer lists. for that i have to change it. 
	// (compile original code with c++11 will work (c++11 is available since vs2013)
	if (doSIFT)
	{
		//*********** SIFT BF Tests ***********//
		//conf = {"SIFT", "SIFT", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "BRIEF", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "ORB", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "SURF", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "BRISK", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "FREAK", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "AKAZE", "BF"};
		conf.detector = "SIFT";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** SIFT FLANN Tests ***********//
		//conf = {"SIFT", "SIFT", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "BRIEF", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "ORB", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "SURF", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "BRISK", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "FREAK", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SIFT", "AKAZE", "FLANN"};
		conf.detector = "SIFT";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doFAST)
	{
		//*********** FAST BF Tests ***********//
		//conf = {"FAST", "SIFT", "BF"};
		conf.detector = "FAST";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "BRIEF", "BF"};
		conf.detector = "FAST";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "ORB", "BF"};
		conf.detector = "FAST";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "SURF", "BF"};
		conf.detector = "FAST";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "BRISK", "BF"};
		conf.detector = "FAST";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "FREAK", "BF"};
		conf.detector = "FAST";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "AKAZE", "BF"};
		conf.detector = "FAST";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** FAST FLANN Tests ***********//
		//conf = {"FAST", "SIFT", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "BRIEF", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "ORB", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "SURF", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "BRISK", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "FREAK", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"FAST", "AKAZE", "FLANN"};
		conf.detector = "FAST";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doGFTT)
	{
		//*********** GFTT BF Tests ***********//
		//conf = {"GFTT", "SIFT", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "BRIEF", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "ORB", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "SURF", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "BRISK", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "FREAK", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "AKAZE", "BF"};
		conf.detector = "GFTT";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** GFTT FLANN Tests ***********//
		//conf = {"GFTT", "SIFT", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "BRIEF", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "ORB", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "SURF", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "BRISK", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "FREAK", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"GFTT", "AKAZE", "FLANN"};
		conf.detector = "GFTT";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doMSER)
	{
		//*********** MSER BF Tests ***********//
		//conf = {"MSER", "SIFT", "BF"};
		conf.detector = "MSER";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "BRIEF", "BF"};
		conf.detector = "MSER";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "ORB", "BF"};
		conf.detector = "MSER";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "SURF", "BF"};
		conf.detector = "MSER";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "BRISK", "BF"};
		conf.detector = "MSER";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "FREAK", "BF"};
		conf.detector = "MSER";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "AKAZE", "BF"};
		conf.detector = "MSER";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** MSER FLANN Tests ***********//
		//conf = {"MSER", "SIFT", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "BRIEF", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "ORB", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "SURF", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "BRISK", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "FREAK", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"MSER", "AKAZE", "FLANN"};
		conf.detector = "MSER";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doDENSE)
	{
		//*********** DENSE BF Tests ***********//
		//conf = {"DENSE", "SIFT", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "BRIEF", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "ORB", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "SURF", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "BRISK", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "FREAK", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "AKAZE", "BF"};
		conf.detector = "DENSE";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** DENSE FLANN Tests ***********//
		//conf = {"DENSE", "SIFT", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "BRIEF", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "ORB", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "SURF", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "BRISK", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "FREAK", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"DENSE", "AKAZE", "FLANN"};
		conf.detector = "DENSE";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doSTAR)
	{
		//*********** STAR BF Tests ***********//
		//conf = {"STAR", "SIFT", "BF"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "BRIEF", "BF"};
		conf.detector = "STAR";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "ORB", "BF"};
		conf.detector = "STAR";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "SURF", "BF"};
		conf.detector = "STAR";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "BRISK", "BF"};
		conf.detector = "STAR";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "FREAK", "BF"};
		conf.detector = "STAR";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "AKAZE", "BF"};
		conf.detector = "STAR";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** STAR FLANN Tests ***********//
		//conf = {"STAR", "SIFT", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "BRIEF", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "ORB", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "SURF", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "BRISK", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "FREAK", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"STAR", "AKAZE", "FLANN"};
		conf.detector = "STAR";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doSURF)
	{
		//*********** SURF BF Tests ***********//
		//conf = {"SURF", "SIFT", "BF"};
		conf.detector = "SURF";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "BRIEF", "BF"};
		conf.detector = "SURF";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "ORB", "BF"};
		conf.detector = "SURF";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "SURF", "BF"};
		conf.detector = "SURF";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "BRISK", "BF"};
		conf.detector = "SURF";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "FREAK", "BF"};
		conf.detector = "SURF";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "AKAZE", "BF"};
		conf.detector = "SURF";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** SURF FLANN Tests ***********//
		//conf = {"SURF", "SIFT", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "BRIEF", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "ORB", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "SURF", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "BRISK", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "FREAK", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"SURF", "AKAZE", "FLANN"};
		conf.detector = "SURF";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doBRISK)
	{
		//*********** BRISK BF Tests ***********//
		//conf = {"BRISK", "SIFT", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "BRIEF", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "ORB", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "SURF", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "BRISK", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "FREAK", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "AKAZE", "BF"};
		conf.detector = "BRISK";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** BRISK FLANN Tests ***********//
		//conf = {"BRISK", "SIFT", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "BRIEF", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "ORB", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "SURF", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "BRISK", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "FREAK", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"BRISK", "AKAZE", "FLANN"};
		conf.detector = "BRISK";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doORB)
	{
		//*********** ORB BF Tests ***********//
		//conf = {"ORB", "SIFT", "BF"};
		conf.detector = "ORB";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "BRIEF", "BF"};
		conf.detector = "ORB";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "ORB", "BF"};
		conf.detector = "ORB";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "SURF", "BF"};
		conf.detector = "ORB";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "BRISK", "BF"};
		conf.detector = "ORB";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "FREAK", "BF"};
		conf.detector = "ORB";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "AKAZE", "BF"};
		conf.detector = "ORB";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** ORB FLANN Tests ***********//
		//conf = {"ORB", "SIFT", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "BRIEF", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "ORB", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "SURF", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "BRISK", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "FREAK", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"ORB", "AKAZE", "FLANN"};
		conf.detector = "ORB";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	if (doAKAZE)
	{
		//*********** AKAZE BF Tests ***********//
		//conf = {"AKAZE", "SIFT", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "SIFT";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "BRIEF", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "BRIEF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "ORB", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "ORB";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "SURF", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "SURF";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "BRISK", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "BRISK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "FREAK", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "FREAK";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "AKAZE", "BF"};
		conf.detector = "AKAZE";
		conf.extractor = "AKAZE";
		conf.matcher = "BF";
		testConfigurations.push_back(conf);

		//*********** AKAZE FLANN Tests ***********//
		//conf = {"AKAZE", "SIFT", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "SIFT";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "BRIEF", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "BRIEF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "ORB", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "ORB";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "SURF", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "SURF";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "BRISK", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "BRISK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "FREAK", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "FREAK";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
		//conf = {"AKAZE", "AKAZE", "FLANN"};
		conf.detector = "AKAZE";
		conf.extractor = "AKAZE";
		conf.matcher = "FLANN";
		testConfigurations.push_back(conf);
	}

	// iterate configurations
	for (vector<Configuration>::iterator config = testConfigurations.begin(); config != testConfigurations.end(); ++config)
	{
		// configure controller
		configure(config->detector, config->extractor, config->matcher);

		cout << "-----------------------------------------------------" << endl;
		cout << "Testing.. [Detector=" << Analyzer::DETECTOR << ", Extractor=" << Analyzer::EXTRACTOR <<
			", Matcher=" << Analyzer::MATCHER << "]" << endl;
		cout << "-----------------------------------------------------" << endl;

		// (re-)create object pattern
		createObjectPattern(objectRgbImage, objectGrayImage);

		int imageCount = images.size();
		for (int i = 1; i < imageCount; i++)
		{
			// load (actual) scene image
			cout << "Loading Scene " << STORAGE_PATH + images.at(i) << endl;
			sceneRgbImageData = imread(STORAGE_PATH + images.at(i));
			if (sceneRgbImageData.empty())
			{
				cout << "Scene image " << i << "/" << imageCount << " cannot be read" << endl;
				return 1;
			}
			cvtColor(sceneRgbImageData, sceneGrayImageData, CV_RGB2GRAY);

			// clone images to clean previous drawings
			sceneRgbImage = sceneRgbImageData.clone();
			sceneGrayImage = sceneGrayImageData.clone();

			// do testruns
			bool shouldQuit = false;
			int isRun = 0;
			do
			{
				// count testrun
				isRun++;

				// do test
				string imageName = images.at(i);
				size_t found = imageName.find_last_of("/\\");
				string fileName = imageName.substr(found + 1);
				int result = displayFunction(sceneRgbImage, sceneGrayImage, fileName);

				// print result
				if (result == 1)
				{
					cout << "Image " << i << "/" << imageCount - 1 << " - Test " << isRun << "/" << doRuns << " - Result: Object found!" << endl;
				}
				else
				{
					cout << "Image " << i << "/" << imageCount - 1 << " - Test " << isRun << "/" << doRuns << " - Result: FAILED!" << endl;
				}


				// continue?
				if (isRun == doRuns)
				{
					shouldQuit = true;
				}
			}
			while (!shouldQuit);
		}
	}

	// restore last state
	isModeObjectDetection(wasObjectDetection);
	isModeTracking(wasTracking);
	isModeStatistics(wasStatistics);

	// success message
	if (MODE_DEBUG)
	{
		cout << "Tests finished successfull!" << endl;
		cout << "Results saved to statistics file: " << STATISTICS_FILE << endl;
	}

	return 1;
}

void Controller::statistics(string key, int value)
{
	stringstream sstr;
	sstr << value;
	stats->add(key, sstr.str());
}

void Controller::statistics(string key, double value)
{
	stringstream sstr;
	sstr << value;
	stats->add(key, sstr.str());
}

void Controller::statistics(string key, long unsigned int value)
{
	stringstream sstr;
	sstr << value;
	stats->add(key, sstr.str());
}

void Controller::statistics(string key, string value)
{
	stats->add(key, value);
}

void Controller::statistics(string key, bool value)
{
	stats->add(key, value ? "true" : "false");
}

#endif
