struct FuseSpotParams {
    width: u32,
    height: u32,
    spot_r: f32,
    spot_g: f32,
    spot_b: f32,
    solidity: f32,
}

@group(0) @binding(0) var<uniform> params: FuseSpotParams;
@group(0) @binding(1) var<storage, read_write> ch0: array<f32>;
@group(0) @binding(2) var<storage, read_write> ch1: array<f32>;
@group(0) @binding(3) var<storage, read_write> ch2: array<f32>;
@group(0) @binding(4) var<storage, read> spot: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }
    let i = x + y * params.width;
    let mix = spot[i] * params.solidity;
    let inv = 1.0 - mix;
    ch0[i] = mix * params.spot_r + inv * ch0[i];
    ch1[i] = mix * params.spot_g + inv * ch1[i];
    ch2[i] = mix * params.spot_b + inv * ch2[i];
}
