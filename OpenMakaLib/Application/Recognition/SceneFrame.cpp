#pragma once
#ifndef OPENMAKAENGINE_SCENEFRAME_CPP
#define OPENMAKAENGINE_SCENEFRAME_CPP

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "SceneFrame.h"
#include "../Controller.h"

using namespace std;
using namespace om;
using namespace cv;

int SceneFrame::MAX_IMAGE_SIZE;
float SceneFrame::IMAGE_SCALE;

SceneFrame::SceneFrame(Mat &rgbInputFrame, Mat &grayInputFrame) {

    if (Controller::MODE_DEBUG) {
        cout << "Creating SceneFrame instance.." << endl;
    }

    rgb = rgbInputFrame;
    gray = grayInputFrame;
    objectPosition = vector<Point2f>(4);

    // get initial size of gray mat
    Size graySize = gray.size();

    // get largest image side
    int maxSize;
    if (graySize.width > graySize.height) {
        maxSize = graySize.width;
    } else {
        maxSize = graySize.height;
    }

    // set initial scale factor
    IMAGE_SCALE = 0.1;

    // calc scale factor via max image size
    while (maxSize / IMAGE_SCALE >= MAX_IMAGE_SIZE) {
        IMAGE_SCALE = IMAGE_SCALE + static_cast<float>(0.1);
    }

    // calc calculative width/height
    float calcWidth = graySize.width / IMAGE_SCALE;
    float calcHeight = graySize.height / IMAGE_SCALE;

    // round to concrete width/height
    int height = static_cast<int>(calcHeight + 0.5);
    int width = static_cast<int>(calcWidth + 0.5);

    // create image holder
    Mat holder;
    holder.create(height, width, CV_8UC1);

    // resize gray image
    try {
        resize(gray, holder, holder.size());
    } catch (Exception &exception) {
        cvError(0, "SceneFrame", "Resizing failed!", __FILE__, __LINE__);
        cout << exception.what() << endl;
    }
    gray = holder;

}

string SceneFrame::getInputResolution() const
{
    ostringstream ss;
    ss << rgb.cols << "x" << rgb.rows;
    return ss.str();
}

string SceneFrame::getProcessingResolution() const
{
    ostringstream ss;
    ss << gray.cols << "x" << gray.rows;
    return ss.str();
}

SceneFrame::~SceneFrame() {
    if (Controller::MODE_DEBUG) {
        cout << "Deleting SceneFrame instance.." << endl;
    }
}

#endif
