#pragma once
#include "camera_model.h"
#include <cmath>
#include <string>

namespace MatchTrack {

    constexpr float DEG2RAD = 0.01745329251994329576923690768489f;
    constexpr float RAD2DEG = 57.295779513082320876798154814105f;

    struct Mat3 {
        float m[3][3];

        static Mat3 Identity() {
            Mat3 r{};
            r.m[0][0] = 1.0f; r.m[1][1] = 1.0f; r.m[2][2] = 1.0f;
            return r;
        }

        static Mat3 FromEuler(float yawDeg, float pitchDeg, float rollDeg) {
            float y = yawDeg * DEG2RAD;
            float p = pitchDeg * DEG2RAD;
            float r = rollDeg * DEG2RAD;

            float cy = std::cos(y), sy = std::sin(y);
            float cp = std::cos(p), sp = std::sin(p);
            float cr = std::cos(r), sr = std::sin(r);

            // Rz(roll) * Rx(pitch) * Ry(yaw)
            Mat3 Ry = { { { cy, 0.0f, sy }, { 0.0f, 1.0f, 0.0f }, { -sy, 0.0f, cy } } };
            Mat3 Rx = { { { 1.0f, 0.0f, 0.0f }, { 0.0f, cp, -sp }, { 0.0f, sp, cp } } };
            Mat3 Rz = { { { cr, -sr, 0.0f }, { sr, cr, 0.0f }, { 0.0f, 0.0f, 1.0f } } };

            return Multiply(Rz, Multiply(Rx, Ry));
        }

        static Mat3 Multiply(const Mat3& a, const Mat3& b) {
            Mat3 res{};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    res.m[i][j] = a.m[i][0] * b.m[0][j] +
                                  a.m[i][1] * b.m[1][j] +
                                  a.m[i][2] * b.m[2][j];
                }
            }
            return res;
        }

        static Mat3 Transpose(const Mat3& a) {
            Mat3 res{};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    res.m[i][j] = a.m[j][i];
                }
            }
            return res;
        }
    };

    struct CameraPose {
        float yaw = 0.0f;     // Degrees (-40 for Left, +40 for Right)
        float pitch = -15.0f; // Degrees (-15 downward tilt)
        float roll = 0.0f;    // Degrees
    };

    struct RigConfiguration {
        std::string name = "DJI Action 4 Dual-Rig 2.7K (80 deg, -15 deg tilt)";
        
        CameraIntrinsics leftCamera = GetDJIAction4_2_7K_Dewarp();
        CameraIntrinsics rightCamera = GetDJIAction4_2_7K_Dewarp();

        CameraPose leftPose = { -40.0f, -15.0f, 0.0f };
        CameraPose rightPose = { 40.0f, -15.0f, 0.0f };

        // Global leveling corrections
        float globalPitchCorrection = 15.0f; // Level the -15° downward tilt
        float globalRollCorrection = 0.0f;
        float globalYawCenter = 0.0f;

        // Panorama framing
        float panoHfov = 130.0f;             // Degrees horizontal panorama FOV
        float verticalCropOffset = 0.0f;     // -1.0 to +1.0
        float blendWidthDeg = 8.0f;          // Overlap seam blend width in degrees
    };

} // namespace MatchTrack

