#include "lut_generator.h"
#include <algorithm>
#include <thread>

namespace MatchTrack {

    RemapLUTs LUTGenerator::Generate(const RigConfiguration& rig, int outWidth, int outHeight) {
        RemapLUTs luts;
        luts.outWidth = outWidth;
        luts.outHeight = outHeight;

        size_t totalPixels = static_cast<size_t>(outWidth) * outHeight;
        luts.mapXLeft.resize(totalPixels, 0.0f);
        luts.mapYLeft.resize(totalPixels, 0.0f);
        luts.maskLeft.resize(totalPixels, 0);

        luts.mapXRight.resize(totalPixels, 0.0f);
        luts.mapYRight.resize(totalPixels, 0.0f);
        luts.maskRight.resize(totalPixels, 0);

        luts.weightLeft.resize(totalPixels, 0.0f);
        luts.weightRight.resize(totalPixels, 0.0f);

        // Global leveling rotation: R_global = Rz(roll) * Rx(pitch) * Ry(yaw)
        Mat3 rGlobal = Mat3::FromEuler(rig.globalYawCenter, rig.globalPitchCorrection, rig.globalRollCorrection);

        // Left & Right camera rotation matrices
        Mat3 rCamLeft = Mat3::FromEuler(rig.leftPose.yaw, rig.leftPose.pitch, rig.leftPose.roll);
        Mat3 rCamRight = Mat3::FromEuler(rig.rightPose.yaw, rig.rightPose.pitch, rig.rightPose.roll);

        float hfovRad = rig.panoHfov * DEG2RAD;
        float fPano = (outWidth * 0.5f) / std::tan(hfovRad * 0.5f);
        float yCenter = outHeight * 0.5f + rig.verticalCropOffset * (outHeight * 0.25f);
        float blendRad = std::max(rig.blendWidthDeg * DEG2RAD, 0.01f);

        // Multi-threaded computation per scanline
        unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> workers;

        auto workerFunc = [&](int startY, int endY) {
            for (int y = startY; y < endY; ++y) {
                float h = (y - yCenter) / fPano;

                for (int x = 0; x < outWidth; ++x) {
                    size_t idx = static_cast<size_t>(y) * outWidth + x;
                    float lambda = (x - (outWidth * 0.5f)) / (outWidth * 0.5f) * (hfovRad * 0.5f);

                    // 1. Ray in cylindrical coords
                    float xg = std::sin(lambda);
                    float yg = h;
                    float zg = std::cos(lambda);

                    float norm = std::sqrt(xg * xg + yg * yg + zg * zg);
                    if (norm > 1e-6f) {
                        xg /= norm; yg /= norm; zg /= norm;
                    }

                    // 2. Level ray
                    float vLevel[3] = {
                        rGlobal.m[0][0] * xg + rGlobal.m[0][1] * yg + rGlobal.m[0][2] * zg,
                        rGlobal.m[1][0] * xg + rGlobal.m[1][1] * yg + rGlobal.m[1][2] * zg,
                        rGlobal.m[2][0] * xg + rGlobal.m[2][1] * yg + rGlobal.m[2][2] * zg
                    };

                    // 3. Project to Left Camera (R_cam^T * vLevel)
                    float xc_l = rCamLeft.m[0][0] * vLevel[0] + rCamLeft.m[1][0] * vLevel[1] + rCamLeft.m[2][0] * vLevel[2];
                    float yc_l = rCamLeft.m[0][1] * vLevel[0] + rCamLeft.m[1][1] * vLevel[1] + rCamLeft.m[2][1] * vLevel[2];
                    float zc_l = rCamLeft.m[0][2] * vLevel[0] + rCamLeft.m[1][2] * vLevel[1] + rCamLeft.m[2][2] * vLevel[2];

                    if (zc_l > 0.05f) {
                        float u_l = rig.leftCamera.fx * (xc_l / zc_l) + rig.leftCamera.cx;
                        float v_l = rig.leftCamera.fy * (yc_l / zc_l) + rig.leftCamera.cy;

                        if (u_l >= 0.0f && u_l < (rig.leftCamera.imageWidth - 1) &&
                            v_l >= 0.0f && v_l < (rig.leftCamera.imageHeight - 1)) {
                            luts.mapXLeft[idx] = u_l;
                            luts.mapYLeft[idx] = v_l;
                            luts.maskLeft[idx] = 1;
                        }
                    }

                    // 4. Project to Right Camera
                    float xc_r = rCamRight.m[0][0] * vLevel[0] + rCamRight.m[1][0] * vLevel[1] + rCamRight.m[2][0] * vLevel[2];
                    float yc_r = rCamRight.m[0][1] * vLevel[0] + rCamRight.m[1][1] * vLevel[1] + rCamRight.m[2][1] * vLevel[2];
                    float zc_r = rCamRight.m[0][2] * vLevel[0] + rCamRight.m[1][2] * vLevel[1] + rCamRight.m[2][2] * vLevel[2];

                    if (zc_r > 0.05f) {
                        float u_r = rig.rightCamera.fx * (xc_r / zc_r) + rig.rightCamera.cx;
                        float v_r = rig.rightCamera.fy * (yc_r / zc_r) + rig.rightCamera.cy;

                        if (u_r >= 0.0f && u_r < (rig.rightCamera.imageWidth - 1) &&
                            v_r >= 0.0f && v_r < (rig.rightCamera.imageHeight - 1)) {
                            luts.mapXRight[idx] = u_r;
                            luts.mapYRight[idx] = v_r;
                            luts.maskRight[idx] = 1;
                        }
                    }

                    // 5. Seamless Smoothstep Blending
                    bool ml = luts.maskLeft[idx] != 0;
                    bool mr = luts.maskRight[idx] != 0;

                    if (ml && !mr) {
                        luts.weightLeft[idx] = 1.0f;
                        luts.weightRight[idx] = 0.0f;
                    } else if (!ml && mr) {
                        luts.weightLeft[idx] = 0.0f;
                        luts.weightRight[idx] = 1.0f;
                    } else if (ml && mr) {
                        float t = (lambda - (-blendRad * 0.5f)) / blendRad;
                        t = std::clamp(t, 0.0f, 1.0f);
                        float wr = 3.0f * t * t - 2.0f * t * t * t;
                        float wl = 1.0f - wr;
                        luts.weightLeft[idx] = wl;
                        luts.weightRight[idx] = wr;
                    }
                }
            }
        };

        int rowsPerThread = (outHeight + numThreads - 1) / numThreads;
        for (unsigned int t = 0; t < numThreads; ++t) {
            int startY = t * rowsPerThread;
            int endY = std::min(outHeight, startY + rowsPerThread);
            if (startY < endY) {
                workers.emplace_back(workerFunc, startY, endY);
            }
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }

        return luts;
    }

} // namespace MatchTrack

