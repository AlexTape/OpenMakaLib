#pragma once
#ifndef OPENMAKAENGINE_OBJECTPATTERN_CPP
#define OPENMAKAENGINE_OBJECTPATTERN_CPP

#include <iostream>
#include "ObjectPattern.h"
#include "../Controller.h"

using namespace std;
using namespace om;

ObjectPattern::ObjectPattern(const Mat &grayImage) {

    if (Controller::MODE_DEBUG) {
        cout << "Creating ObjectPattern instance.." << endl;
    }

    image = grayImage;

    points2d = vector<Point2f>(4);
    points3d = vector<Point3f>(4);

    build();
}

ObjectPattern::~ObjectPattern() {
    if (Controller::MODE_DEBUG) {
        cout << "Deleting ObjectPattern instance.." << endl;
    }
}

void ObjectPattern::build() {

    // set size
    size = Size(image.cols, image.rows);

    // set normalized dimensions
    float maximumSize = max(size.width, size.height);
    float widthUnit = size.width / maximumSize;
    float heightUnit = size.height / maximumSize;

    // set points2d
    points2d[0] = Point2f(0, 0);
    points2d[1] = Point2f(size.width, 0);
    points2d[2] = Point2f(size.width, size.height);
    points2d[3] = Point2f(0, size.height);

    // set points3d
    points3d[0] = Point3f(-widthUnit, -heightUnit, 0);
    points3d[1] = Point3f(widthUnit, -heightUnit, 0);
    points3d[2] = Point3f(widthUnit, heightUnit, 0);
    points3d[3] = Point3f(-widthUnit, heightUnit, 0);
}

#endif
