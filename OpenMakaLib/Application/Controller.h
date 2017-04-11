#ifndef OPENMAKAENGINE_CONTROLLER_H
#define OPENMAKAENGINE_CONTROLLER_H

#include <string>

#include <opencv2/core/core.hpp>

#include "Recognition/Analyzer.h"
#include "Recognition/SceneFrame.h"
#include "Tracker/Tracker.h"
#include "Helper/Timer.h"
#include "Helper/Statistics.h"

#define ControllerTAG "OpenMaka::Controller"

namespace om
{
	class Controller
	{
	public:

		virtual ~Controller();

		static Controller* getInstance();

		int initialize(Mat& frame, string storagePath, string configFile) const;

		int displayFunction(Mat& mRgbaFrame, Mat& mGrayFrame, string imageName = "nill");

		int setDetector(string type) const;

		int setExtractor(string type) const;

		int setMatcher(string type) const;

		static void isModeObjectDetection(bool isActive);

		static void isModeTracking(bool isActive);

		static void isModeOpenGL(bool isActive);

		static bool createObjectPattern(Mat& rgb, Mat& gray);

		bool configure(string detector, string extractor, string matcher) const;
		int test(vector<string> images, int test, int quantifier);
		//int test(vector<std::basic_string<char>> images, int test, int quantifier);

		Timer* clock;
		Timer* timer;
		SceneFrame* sceneFrame;

		static bool isInitialized;

		static Size FRAME_SIZE;
		static int MAX_IMAGE_SIZE;
		static int IMAGE_SCALE;
		static string STATISTICS_FILE;

		static string STORAGE_PATH;
		static string CONFIG_FILE;
		static string DEFAULT_OBJECT_IMAGE;

		static void statistics(string key, int value);

		static void statistics(string key, double value);

		static void statistics(string key, bool value);

		static void statistics(string key, long unsigned int value);

		static void statistics(string key, string value);

		static bool MODE_STATISTICS;
		static bool MODE_DEBUG;
		static bool MODE_USE_WINDOWS;
		static bool MODE_SAVE_RESULT_FRAMES;

		static void isModeDebug(bool isActive);

		static void isModeStatistics(bool isActive);

		typedef struct Configuration
		{
			string detector;
			string extractor;
			string matcher;
		} Configuration;

	private:
		static Controller* inst_;

		Controller();

		static bool MODE_OBJECT_DETECTION;
		static bool MODE_TRACKING;
		static bool MODE_OPENGL;

		bool STATE_OBJECT_FOUND;
		bool STATE_TRACKING_OBJECT;
		bool STATE_DISPLAY_OPENGL;

		static Analyzer* analyzer; // Analyzer object
		static Tracker* tracker; // Tracker object
		static Statistics* stats; // Statistics object
	};
}

#endif
