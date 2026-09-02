#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include "../core/rig_geometry.h"
#include <vector>
#include <string>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

namespace MatchTrack {

    struct CBufferRigParams {
        float matLevelLeft[16];   // 4x4 matrix
        float matLevelRight[16];  // 4x4 matrix
        float leftIntrinsics[4];  // fx, fy, cx, cy
        float rightIntrinsics[4]; // fx, fy, cx, cy
        float panoParams[4];      // outWidth, outHeight, hfovRad, blendRad
        float framingParams[4];   // fPano, yCenter, verticalOffset, padding
    };

    class D3D11Stitcher {
    public:
        D3D11Stitcher();
        ~D3D11Stitcher();

        bool Initialize(HWND hWnd, int panoWidth, int panoHeight);
        void ResizeOutput(int width, int height);

        void UpdateRigParams(const RigConfiguration& rig);
        void UploadCameraFrames(const uint8_t* leftBGR, int leftW, int leftH,
                                const uint8_t* rightBGR, int rightW, int rightH);

        void DispatchCompute();
        void Present();

        ID3D11ShaderResourceView* GetOutputSRV() const { return m_outputSRV.Get(); }
        ID3D11Device* GetDevice() const { return m_device.Get(); }
        ID3D11DeviceContext* GetContext() const { return m_context.Get(); }

    private:
        bool CompileComputeShader(const std::wstring& shaderPath);
        void CreateTexture(int width, int height, DXGI_FORMAT format, UINT bindFlags,
                           ID3D11Texture2D** ppTex, ID3D11ShaderResourceView** ppSRV, ID3D11UnorderedAccessView** ppUAV);

        ComPtr<ID3D11Device> m_device;
        ComPtr<ID3D11DeviceContext> m_context;
        ComPtr<IDXGISwapChain> m_swapChain;
        ComPtr<ID3D11RenderTargetView> m_renderTargetView;

        ComPtr<ID3D11ComputeShader> m_computeShader;
        ComPtr<ID3D11Buffer> m_constantBuffer;
        ComPtr<ID3D11SamplerState> m_linearSampler;

        // Textures
        ComPtr<ID3D11Texture2D> m_texLeft;
        ComPtr<ID3D11ShaderResourceView> m_srvLeft;

        ComPtr<ID3D11Texture2D> m_texRight;
        ComPtr<ID3D11ShaderResourceView> m_srvRight;

        ComPtr<ID3D11Texture2D> m_texOutput;
        ComPtr<ID3D11ShaderResourceView> m_outputSRV;
        ComPtr<ID3D11UnorderedAccessView> m_outputUAV;

        int m_panoWidth = 3840;
        int m_panoHeight = 1080;
        int m_camWidth = 2720;
        int m_camHeight = 1530;
    };

} // namespace MatchTrack

