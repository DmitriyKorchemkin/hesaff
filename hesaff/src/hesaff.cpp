#include "hesaff/hesaff.h"

using namespace cv;

AffineHessianDetector::AffineHessianDetector(const cv::Mat &image,
                                             const PyramidParams &par,
                                             const AffineShapeParams &ap,
                                             const SIFTDescriptorParams &sp)
    : HessianDetector(par), AffineShape(ap), image(image), sift(sp)
{
    this->setHessianKeypointCallback(this);
    this->setAffineShapeCallback(this);
}

void AffineHessianDetector::onHessianKeypointDetected(const cv::Mat &blur,
                                                      float x, float y, float s,
                                                      float pixelDistance,
                                                      int type, float response)
{
    g_numberOfPoints++;
    findAffineShape(blur, x, y, s, pixelDistance, type, response);
}

void AffineHessianDetector::onAffineShapeFound(const cv::Mat &blur, float x,
                                               float y, float s,
                                               float pixelDistance, float a11,
                                               float a12, float a21, float a22,
                                               int type, float response,
                                               int iters)
{
    // convert shape into a up is up frame
    rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);

    // now sample the patch
    if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
    {
        // compute SIFT
        sift.computeSiftDescriptor(this->patch);
        // store the keypoint
        keys.push_back(Keypoint());
        Keypoint &k = keys.back();
        k.x = x;
        k.y = y;
        k.s = s;
        k.a11 = a11;
        k.a12 = a12;
        k.a21 = a21;
        k.a22 = a22;
        k.response = response;
        k.type = type;
        for (int i = 0; i < 128; i++)
            k.desc[i] = (unsigned char)sift.vec[i];
        // debugging stuff
        if (0)
        {
            std::cout << "x: " << x << ", y: " << y << ", s: " << s
                      << ", pd: " << pixelDistance << ", a11: " << a11
                      << ", a12: " << a12 << ", a21: " << a21
                      << ", a22: " << a22 << ", t: " << type
                      << ", r: " << response << std::endl;
            for (size_t i = 0; i < sift.vec.size(); i++)
                std::cout << " " << sift.vec[i];
            std::cout << std::endl;
        }
        g_numberOfAffinePoints++;
    }
}

void AffineHessianDetector::exportKeypoints(std::ostream &out)
{
    out << 128 << std::endl;
    out << keys.size() << std::endl;
    for (size_t i = 0; i < keys.size(); i++)
    {
        Keypoint &k = keys[i];

        float sc = AffineShape::par.mrSize * k.s;
        cv::Mat A = (cv::Mat_<float>(2, 2) << k.a11, k.a12, k.a21, k.a22);
        SVD svd(A, SVD::FULL_UV);

        float *d = (float *)svd.w.data;
        d[0] = 1.0f / (d[0] * d[0] * sc * sc);
        d[1] = 1.0f / (d[1] * d[1] * sc * sc);

        A = svd.u * cv::Mat::diag(svd.w) * svd.u.t();

        out << k.x << " " << k.y << " " << A.at<float>(0, 0) << " "
            << A.at<float>(0, 1) << " " << A.at<float>(1, 1);
        for (size_t i = 0; i < 128; i++)
            out << " " << int(k.desc[i]);
        out << std::endl;
    }
}
