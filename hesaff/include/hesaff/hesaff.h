#ifndef HESAFF_H
#define HESAFF_H
#include "hesaff/affine.h"
#include "hesaff/helpers.h"
#include "hesaff/pyramid.h"
#include "hesaff/siftdesc.h"

struct HessianAffineParams
{
    float threshold;
    int max_iter;
    float desc_factor;
    int patch_size;
    bool verbose;
    HessianAffineParams()
    {
        threshold = 16.0f / 3.0f;
        max_iter = 16;
        desc_factor = 3.0f * sqrt(3.0f);
        patch_size = 41;
        verbose = false;
    }
};

struct Keypoint
{
    float x, y, s;
    float a11, a12, a21, a22;
    float response;
    int type;
    unsigned char desc[128];
};

struct AffineHessianDetector : public HessianDetector,
                               AffineShape,
                               HessianKeypointCallback,
                               AffineShapeCallback
{
    const cv::Mat image;
    SIFTDescriptor sift;
    std::vector<Keypoint> keys;
    int g_numberOfPoints = 0;
    int g_numberOfAffinePoints = 0;

  public:
    AffineHessianDetector(const cv::Mat &image, const PyramidParams &par,
                          const AffineShapeParams &ap,
                          const SIFTDescriptorParams &sp);

    void onHessianKeypointDetected(const cv::Mat &blur, float x, float y,
                                   float s, float pixelDistance, int type,
                                   float response);

    void onAffineShapeFound(const cv::Mat &blur, float x, float y, float s,
                            float pixelDistance, float a11, float a12,
                            float a21, float a22, int type, float response,
                            int iters);

    void exportKeypoints(std::ostream &out);
};

#endif
