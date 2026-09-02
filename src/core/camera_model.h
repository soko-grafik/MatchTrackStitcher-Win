#pragma once
#include <string>
#include <vector>
#include <cmath>

namespace MatchTrack {

    struct CameraIntrinsics {
        std::string name = "DJI Action 4 2.7K Standard (Dewarp)";
        int imageWidth = 2720;
        int imageHeight = 1530;
        
        // Rectilinear focal length for DJI Action 4 in Dewarp/Standard mode (approx 93 deg HFOV)
        float fx = 1950.0f;
        float fy = 1950.0f;
        float cx = 1360.0f;
        float cy = 765.0f;
        
        // Lens distortion (0.0 for Dewarp mode since DJI DSP already dewarps)
        float k1 = 0.0f;
        float k2 = 0.0f;
        float k3 = 0.0f;
        float k4 = 0.0f;
    };

    inline CameraIntrinsics GetDJIAction4_2_7K_Dewarp() {
        CameraIntrinsics intr;
        intr.name = "DJI Action 4 2.7K (16:9) Standard/Dewarp";
        intr.imageWidth = 2720;
        intr.imageHeight = 1530;
        intr.fx = 1315.0f;
        intr.fy = 1315.0f;
        intr.cx = 1360.0f;
        intr.cy = 765.0f;
        return intr;
    }

    inline CameraIntrinsics GetDJIAction4_4K_Dewarp() {
        CameraIntrinsics intr;
        intr.name = "DJI Action 4 4K (16:9) Standard/Dewarp";
        intr.imageWidth = 3840;
        intr.imageHeight = 2160;
        intr.fx = 2750.0f;
        intr.fy = 2750.0f;
        intr.cx = 1920.0f;
        intr.cy = 1080.0f;
        return intr;
    }

} // namespace MatchTrack

