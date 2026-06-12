struct UpsampleParams {
    in_width: u32,
    in_height: u32,
    k: u32,
    mat_n: u32,
    out_width: u32,
    out_height: u32,
}

@group(0) @binding(0) var<uniform> params: UpsampleParams;
@group(0) @binding(1) var<storage, read> weights_quarter: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn mirror_i32(offset_in: i32, len: i32) -> u32 {
    var offset = offset_in;
    while true {
        if offset < 0 {
            offset = -(offset + 1);
        } else if offset >= len {
            offset = (-(offset + 1)) + len * 2;
        } else {
            return u32(offset);
        }
    }
    return 0u;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.out_width || y >= params.out_height {
        return;
    }

    let K = params.k;
    let mat_n = params.mat_n;
    let ref_x = x / K;
    let ref_y = y / K;
    let rem_x = x % K;
    let rem_y = y % K;
    let mat_x = min(rem_x, K - 1u - rem_x);
    let mat_y = min(rem_y, K - 1u - rem_y);
    let flip_h = rem_x >= mat_n;
    let flip_v = rem_y >= mat_n;
    let kernel_idx = mat_y * mat_n + mat_x;

    var sum = 0.0;
    var vmin = 1e38;
    var vmax = -1e38;
    for (var iy = 0u; iy < 5u; iy = iy + 1u) {
        let ky = select(iy, 4u - iy, flip_v);
        for (var ix = 0u; ix < 5u; ix = ix + 1u) {
            let kx = select(ix, 4u - ix, flip_h);
            let sx = i32(ref_x) + i32(ix) - 2;
            let sy = i32(ref_y) + i32(iy) - 2;
            let mx = mirror_i32(sx, i32(params.in_width));
            let my = mirror_i32(sy, i32(params.in_height));
            let sample = input[mx + my * params.in_width];
            let w = weights_quarter[kernel_idx * 25u + ky * 5u + kx];
            sum = sum + w * sample;
            vmin = min(vmin, sample);
            vmax = max(vmax, sample);
        }
    }

    let out_idx = x + y * params.out_width;
    if (vmin != vmin) {
        output[out_idx] = bitcast<f32>(0x7fc00000u);
    } else {
        output[out_idx] = clamp(sum, vmin, vmax);
    }
}
