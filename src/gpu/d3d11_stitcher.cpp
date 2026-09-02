#include "d3d11_stitcher.h"
#include <iostream>

namespace MatchTrack {

    D3D11Stitcher::D3D11Stitcher() = default;
    D3D11Stitcher::~D3D11Stitcher() = default;

    bool D3D11Stitcher::Initialize(HWND hWnd, int panoWidth, int panoHeight) {
        m_panoWidth = panoWidth;
        m_panoHeight = panoHeight;

        DXGI_SWAP_CHAIN_DESC scd = {};
        scd.BufferCount = 1;
        scd.BufferDesc.Width = panoWidth;
        scd.BufferDesc.Height = panoHeight;
        scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        scd.BufferDesc.RefreshRate.Numerator = 60;
        scd.BufferDesc.RefreshRate.Denominator = 1;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.OutputWindow = hWnd;
        scd.SampleDesc.Count = 1;
        scd.Windowed = TRUE;

        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
        D3D_FEATURE_LEVEL featureLevel;

        HRESULT hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            0, featureLevels, 1, D3D11_SDK_VERSION,
            &scd, m_swapChain.GetAddressOf(),
            m_device.GetAddressOf(), &featureLevel,
            m_context.GetAddressOf()
        );

        if (FAILED(hr)) return false;

        // Render target
        ComPtr<ID3D11Texture2D> backBuffer;
        m_swapChain->GetBuffer(0, IID_PPV_ARGS(backBuffer.GetAddressOf()));
        m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, m_renderTargetView.GetAddressOf());

        // Constant Buffer for Rig Parameters
        D3D11_BUFFER_DESC bd = {};
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.ByteWidth = sizeof(CBufferRigParams);
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        m_device->CreateBuffer(&bd, nullptr, m_constantBuffer.GetAddressOf());

        // Linear Sampler
        D3D11_SAMPLER_DESC sampDesc = {};
        sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        m_device->CreateSamplerState(&sampDesc, m_linearSampler.GetAddressOf());

        // Camera textures (2.7K: 2720x1530 RGBA)
        CreateTexture(m_camWidth, m_camHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_SHADER_RESOURCE,
                      m_texLeft.GetAddressOf(), m_srvLeft.GetAddressOf(), nullptr);

        CreateTexture(m_camWidth, m_camHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_SHADER_RESOURCE,
                      m_texRight.GetAddressOf(), m_srvRight.GetAddressOf(), nullptr);

        // Output Panorama texture (UAV + SRV)
        CreateTexture(m_panoWidth, m_panoHeight, DXGI_FORMAT_R8G8B8A8_UNORM,
                      D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
                      m_texOutput.GetAddressOf(), m_outputSRV.GetAddressOf(), m_outputUAV.GetAddressOf());

        // Compile Compute Shader
        CompileComputeShader(L"shaders/stitch_compute.hlsl");
        return true;
    }

    void D3D11Stitcher::CreateTexture(int width, int height, DXGI_FORMAT format, UINT bindFlags,
                                      ID3D11Texture2D** ppTex, ID3D11ShaderResourceView** ppSRV, ID3D11UnorderedAccessView** ppUAV) {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = width;
        td.Height = height;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = format;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = bindFlags;
        td.CPUAccessFlags = 0;

        m_device->CreateTexture2D(&td, nullptr, ppTex);

        if (ppSRV && (bindFlags & D3D11_BIND_SHADER_RESOURCE)) {
            m_device->CreateShaderResourceView(*ppTex, nullptr, ppSRV);
        }
        if (ppUAV && (bindFlags & D3D11_BIND_UNORDERED_ACCESS)) {
            m_device->CreateUnorderedAccessView(*ppTex, nullptr, ppUAV);
        }
    }

    bool D3D11Stitcher::CompileComputeShader(const std::wstring& shaderPath) {
        ComPtr<ID3DBlob> csBlob;
        ComPtr<ID3DBlob> errorBlob;

        HRESULT hr = D3DCompileFromFile(
            shaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
            "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
            csBlob.GetAddressOf(), errorBlob.GetAddressOf()
        );

        if (FAILED(hr)) {
            if (errorBlob) {
                OutputDebugStringA((char*)errorBlob->GetBufferPointer());
            }
            return false;
        }

        m_device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, m_computeShader.GetAddressOf());
        return true;
    }

    void D3D11Stitcher::UpdateRigParams(const RigConfiguration& rig) {
        Mat3 rGlobal = Mat3::FromEuler(rig.globalYawCenter, rig.globalPitchCorrection, rig.globalRollCorrection);
        Mat3 rCamLeft = Mat3::FromEuler(rig.leftPose.yaw, rig.leftPose.pitch, rig.leftPose.roll);
        Mat3 rCamRight = Mat3::FromEuler(rig.rightPose.yaw, rig.rightPose.pitch, rig.rightPose.roll);

        // Combined matrices
        Mat3 combinedL = Mat3::Multiply(Mat3::Transpose(rCamLeft), rGlobal);
        Mat3 combinedR = Mat3::Multiply(Mat3::Transpose(rCamRight), rGlobal);

        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(m_context->Map(m_constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            CBufferRigParams* cb = static_cast<CBufferRigParams*>(mapped.pData);

            // 4x4 left matrix
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    cb->matLevelLeft[r * 4 + c] = combinedL.m[r][c];

            // 4x4 right matrix
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    cb->matLevelRight[r * 4 + c] = combinedR.m[r][c];

            cb->leftIntrinsics[0] = rig.leftCamera.fx;
            cb->leftIntrinsics[1] = rig.leftCamera.fy;
            cb->leftIntrinsics[2] = rig.leftCamera.cx;
            cb->leftIntrinsics[3] = rig.leftCamera.cy;

            cb->rightIntrinsics[0] = rig.rightCamera.fx;
            cb->rightIntrinsics[1] = rig.rightCamera.fy;
            cb->rightIntrinsics[2] = rig.rightCamera.cx;
            cb->rightIntrinsics[3] = rig.rightCamera.cy;

            float hfovRad = rig.panoHfov * DEG2RAD;
            cb->panoParams[0] = (float)m_panoWidth;
            cb->panoParams[1] = (float)m_panoHeight;
            cb->panoParams[2] = hfovRad;
            cb->panoParams[3] = std::max(rig.blendWidthDeg * DEG2RAD, 0.01f);

            float fPano = (m_panoWidth * 0.5f) / std::tan(hfovRad * 0.5f);
            float yCenter = m_panoHeight * 0.5f + rig.verticalCropOffset * (m_panoHeight * 0.25f);
            cb->framingParams[0] = fPano;
            cb->framingParams[1] = yCenter;
            cb->framingParams[2] = rig.verticalCropOffset;
            cb->framingParams[3] = 0.0f;

            m_context->Unmap(m_constantBuffer.Get(), 0);
        }
    }

    void D3D11Stitcher::DispatchCompute() {
        if (!m_computeShader) return;

        m_context->CSSetShader(m_computeShader.Get(), nullptr, 0);
        m_context->CSSetConstantBuffers(0, 1, m_constantBuffer.GetAddressOf());
        m_context->CSSetSamplers(0, 1, m_linearSampler.GetAddressOf());

        ID3D11ShaderResourceView* srvs[] = { m_srvLeft.Get(), m_srvRight.Get() };
        m_context->CSSetShaderResources(0, 2, srvs);

        ID3D11UnorderedAccessView* uavs[] = { m_outputUAV.Get() };
        m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);

        // Dispatch 16x16 thread groups
        UINT dispatchX = (m_panoWidth + 15) / 16;
        UINT dispatchY = (m_panoHeight + 15) / 16;
        m_context->Dispatch(dispatchX, dispatchY, 1);

        // Unbind UAV & SRV
        ID3D11UnorderedAccessView* nullUAV[] = { nullptr };
        m_context->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);

        ID3D11ShaderResourceView* nullSRV[] = { nullptr, nullptr };
        m_context->CSSetShaderResources(0, 2, nullSRV);
    }

    void D3D11Stitcher::Present() {
        if (m_swapChain) {
            m_swapChain->Present(1, 0);
        }
    }

} // namespace MatchTrack

