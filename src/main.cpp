#include <windows.h>
#include <iostream>
#include "core/rig_geometry.h"
#include "gpu/d3d11_stitcher.h"
#include "video/video_pipeline.h"

using namespace MatchTrack;

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // 1. Register Window Class
    const wchar_t CLASS_NAME[] = L"MatchTrackStitcherWindowClass";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

    RegisterClass(&wc);

    // 2. Create Window (32:9 Aspect Ratio: e.g. 1280x360 window)
    HWND hWnd = CreateWindowEx(
        0, CLASS_NAME, L"MatchTrack-Stitcher [Native C++ / DirectX 11 / RTX 3060 NVENC]",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 420,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!hWnd) return 0;

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // 3. Initialize Rig & D3D11 GPU Stitcher
    RigConfiguration rig;
    rig.leftCamera = GetDJIAction4_2_7K_Dewarp();
    rig.rightCamera = GetDJIAction4_2_7K_Dewarp();
    rig.leftPose = { -40.0f, -15.0f, 0.0f };
    rig.rightPose = { 40.0f, -15.0f, 0.0f };
    rig.globalPitchCorrection = 15.0f; // Level 15° tilt
    rig.panoHfov = 130.0f;

    D3D11Stitcher stitcher;
    if (stitcher.Initialize(hWnd, 3840, 1080)) {
        stitcher.UpdateRigParams(rig);
    }

    // 4. Main Event & Render Loop
    MSG msg = {};
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            // Render 32:9 frame on GPU
            stitcher.DispatchCompute();
            stitcher.Present();
            Sleep(16); // ~60 FPS
        }
    }

    return (int)msg.wParam;
}

