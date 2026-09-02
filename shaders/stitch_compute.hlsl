// DirectCompute HLSL Shader for Real-time 32:9 Panorama Video Stitching
// Executes on NVIDIA RTX GPU in Direct3D 11

Texture2D<float4> g_TexLeft : register(t0);
Texture2D<float4> g_TexRight : register(t1);
SamplerState g_SamplerLinear : register(s0);

RWTexture2D<float4> g_OutputPano : register(u0);

cbuffer RigParams : register(b0)
{
    float4x4 g_MatLevelLeft;  // Combined R_cam_l^T * R_global matrix
    float4x4 g_MatLevelRight; // Combined R_cam_r^T * R_global matrix
    float4 g_LeftIntrinsics;   // (fx, fy, cx, cy)
    float4 g_RightIntrinsics;  // (fx, fy, cx, cy)
    float4 g_PanoParams;       // (outWidth, outHeight, hfovRad, blendRad)
    float4 g_FramingParams;    // (fPano, yCenter, verticalOffset, padding)
};

[numthreads(16, 16, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;

    uint outWidth = (uint)g_PanoParams.x;
    uint outHeight = (uint)g_PanoParams.y;

    if (x >= outWidth || y >= outHeight)
        return;

    float hfovRad = g_PanoParams.z;
    float blendRad = g_PanoParams.w;
    float fPano = g_FramingParams.x;
    float yCenter = g_FramingParams.y;

    float lambda = ((float)x - (outWidth * 0.5f)) / (outWidth * 0.5f) * (hfovRad * 0.5f);
    float h = ((float)y - yCenter) / fPano;

    // 1. Ray in world cylindrical frame
    float3 rayWorld = normalize(float3(sin(lambda), h, cos(lambda)));

    // 2. Transform into Left Camera Space
    float3 rayLeft = mul((float3x3)g_MatLevelLeft, rayWorld);
    float4 colorLeft = float4(0, 0, 0, 0);
    float maskLeft = 0.0f;

    if (rayLeft.z > 0.05f)
    {
        float2 uv_l;
        uv_l.x = (g_LeftIntrinsics.x * (rayLeft.x / rayLeft.z) + g_LeftIntrinsics.z) / 2720.0f;
        uv_l.y = (g_LeftIntrinsics.y * (rayLeft.y / rayLeft.z) + g_LeftIntrinsics.w) / 1530.0f;

        if (uv_l.x >= 0.0f && uv_l.x <= 1.0f && uv_l.y >= 0.0f && uv_l.y <= 1.0f)
        {
            colorLeft = g_TexLeft.SampleLevel(g_SamplerLinear, uv_l, 0);
            maskLeft = 1.0f;
        }
    }

    // 3. Transform into Right Camera Space
    float3 rayRight = mul((float3x3)g_MatLevelRight, rayWorld);
    float4 colorRight = float4(0, 0, 0, 0);
    float maskRight = 0.0f;

    if (rayRight.z > 0.05f)
    {
        float2 uv_r;
        uv_r.x = (g_RightIntrinsics.x * (rayRight.x / rayRight.z) + g_RightIntrinsics.z) / 2720.0f;
        uv_r.y = (g_RightIntrinsics.y * (rayRight.y / rayRight.z) + g_RightIntrinsics.w) / 1530.0f;

        if (uv_r.x >= 0.0f && uv_r.x <= 1.0f && uv_r.y >= 0.0f && uv_r.y <= 1.0f)
        {
            colorRight = g_TexRight.SampleLevel(g_SamplerLinear, uv_r, 0);
            maskRight = 1.0f;
        }
    }

    // 4. Smoothstep Blending
    float4 finalColor = float4(0, 0, 0, 1);

    if (maskLeft > 0.5f && maskRight < 0.5f)
    {
        finalColor = colorLeft;
    }
    else if (maskLeft < 0.5f && maskRight > 0.5f)
    {
        finalColor = colorRight;
    }
    else if (maskLeft > 0.5f && maskRight > 0.5f)
    {
        float t = saturate((lambda - (-blendRad * 0.5f)) / blendRad);
        float wRight = 3.0f * t * t - 2.0f * t * t * t;
        float wLeft = 1.0f - wRight;
        finalColor = colorLeft * wLeft + colorRight * wRight;
    }

    g_OutputPano[uint2(x, y)] = finalColor;
}

