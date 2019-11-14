/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#include <fstream>
#include <iostream>

#include "hesaff/hesaff.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        Mat tmp = imread(argv[1]);
        Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));

        float *out = image.ptr<float>(0);
        unsigned char *in = tmp.ptr<unsigned char>(0);

        for (size_t i = tmp.rows * tmp.cols; i > 0; i--)
        {
            *out = (float(in[0]) + in[1] + in[2]) / 3.0f;
            out++;
            in += 3;
        }

        HessianAffineParams par;
        double t1 = 0;
        {
            // copy params
            PyramidParams p;
            p.threshold = par.threshold;

            AffineShapeParams ap;
            ap.maxIterations = par.max_iter;
            ap.patchSize = par.patch_size;
            ap.mrSize = par.desc_factor;

            SIFTDescriptorParams sp;
            sp.patchSize = par.patch_size;

            AffineHessianDetector detector(image, p, ap, sp);
            t1 = getTime();
            detector.g_numberOfPoints = 0;
            detector.detectPyramidKeypoints(image);
            cout << "Detected " << detector.g_numberOfPoints
                 << " keypoints and " << detector.g_numberOfAffinePoints
                 << " affine shapes in " << getTime() - t1 << " sec." << endl;

            char suffix[] = ".hesaff.sift";
            int len = strlen(argv[1]) + strlen(suffix) + 1;
            char buf[len];
            snprintf(buf, len, "%s%s", argv[1], suffix);
            buf[len - 1] = 0;
            ofstream out(buf);
            detector.exportKeypoints(out);
        }
    }
    else
    {
        printf("\nUsage: hesaff image_name.ppm\nDetects Hessian Affine points "
               "and describes them using SIFT descriptor.\nThe detector "
               "assumes that the vertical orientation is preserved.\n\n");
    }
}
