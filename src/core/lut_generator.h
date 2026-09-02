#pragma once
#include "rig_geometry.h"
#include <vector>

namespace MatchTrack {

    struct RemapLUTs {
        int outWidth = 0;
        int outHeight = 0;

        // Pixel mapping arrays: size = outWidth * outHeight
        std::vector<float> mapXLeft;
        std::vector<float> mapYLeft;
        std::vector<uint8_t> maskLeft;

        std::vector<float> mapXRight;
        std::vector<float> mapYRight;
        std::vector<uint8_t> maskRight;

        std::vector<float> weightLeft;
        std::vector<float> weightRight;
    };

    class LUTGenerator {
    public:
        static RemapLUTs Generate(const RigConfiguration& rig, int outWidth, int outHeight);
    };

} // namespace MatchTrack

