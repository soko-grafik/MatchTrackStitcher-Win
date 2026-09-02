#pragma once
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <windows.h>
#include "../core/rig_geometry.h"

namespace MatchTrack {

    struct VideoMetadata {
        int width = 0;
        int height = 0;
        double fps = 60.0;
        int64_t totalFrames = 0;
        double durationSec = 0.0;
    };

    class VideoPipeline {
    public:
        VideoPipeline();
        ~VideoPipeline();

        bool OpenVideos(const std::string& pathLeft, const std::string& pathRight);
        void Close();

        VideoMetadata GetLeftMetadata() const { return m_metaLeft; }
        VideoMetadata GetRightMetadata() const { return m_metaRight; }

        int CalculateAudioSyncOffset();

        bool RenderBatch(const std::string& outputPath,
                         const RigConfiguration& rig,
                         int outWidth, int outHeight,
                         const std::string& codec, int bitrateMbps,
                         int frameOffsetRight,
                         std::function<void(int64_t current, int64_t total, double fps, double eta)> progressCallback);

        void CancelRender();

    private:
        VideoMetadata ProbeMetadata(const std::string& filepath);

        std::string m_pathLeft;
        std::string m_pathRight;
        VideoMetadata m_metaLeft;
        VideoMetadata m_metaRight;

        std::atomic<bool> m_cancelRequested{ false };
    };

} // namespace MatchTrack

