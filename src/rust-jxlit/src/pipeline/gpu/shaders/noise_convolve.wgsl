struct NoiseConvolveParams {
    frame_width: u32,
    frame_height: u32,
    group_dim: u32,
    groups_per_row: u32,
    num_groups: u32,
}

struct GroupMeta {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    stride: u32,
    raw_offset: u32,
}

@group(0) @binding(0) var<uniform> params: NoiseConvolveParams;
@group(0) @binding(1) var<storage, read> group_meta: array<GroupMeta>;
@group(0) @binding(2) var<storage, read> raw_noise: array<f32>;
@group(0) @binding(3) var<storage, read_write> convolved: array<f32>;

fn group_index_for(px: u32, py: u32) -> u32 {
    let gx = px / params.group_dim;
    let gy = py / params.group_dim;
    return gy * params.groups_per_row + gx;
}

fn read_raw_in_group(group_idx: u32, lx: u32, ly: u32, ch: u32) -> f32 {
    let gmeta = group_meta[group_idx];
    if lx >= gmeta.width || ly >= gmeta.height {
        return 0.0;
    }
    let plane = gmeta.stride * gmeta.height;
    let idx = gmeta.raw_offset + ch * plane + ly * gmeta.stride + lx;
    return raw_noise[idx];
}

fn read_raw_global(px: u32, py: u32, ch: u32) -> f32 {
    if px >= params.frame_width || py >= params.frame_height {
        return 0.0;
    }
    let gidx = group_index_for(px, py);
    let gmeta = group_meta[gidx];
    let lx = px - gmeta.x0;
    let ly = py - gmeta.y0;
    return read_raw_in_group(gidx, lx, ly, ch);
}

fn horiz_sample(sx: i32) -> u32 {
    if sx < 0 {
        return u32(-sx - 1);
    }
    if sx >= i32(params.frame_width) {
        let over = sx - i32(params.frame_width);
        if over <= 1 {
            return params.frame_width - 1u;
        }
        let lx = i32(params.frame_width) - 1 - (over - 1);
        return u32(max(lx, 0));
    }
    return u32(sx);
}

fn vert_sample(sy: i32) -> u32 {
    if sy < 0 {
        return u32(-sy - 1);
    }
    if sy >= i32(params.frame_height) {
        let over = sy - i32(params.frame_height);
        if over <= 1 {
            return params.frame_height - 1u;
        }
        let ly = i32(params.frame_height) - 1 - (over - 1);
        return u32(max(ly, 0));
    }
    return u32(sy);
}

fn sample_raw(sx: i32, sy: i32, ch: u32) -> f32 {
    let px = horiz_sample(sx);
    let py = vert_sample(sy);
    return read_raw_global(px, py, ch);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= params.frame_width || py >= params.frame_height {
        return;
    }

    let plane = params.frame_width * params.frame_height;
    let out_base = py * params.frame_width + px;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        var sum = 0.0;
        for (var dy = 0; dy < 5; dy = dy + 1) {
            for (var dx = 0; dx < 5; dx = dx + 1) {
                let sx = i32(px) + dx - 2;
                let sy = i32(py) + dy - 2;
                sum += sample_raw(sx, sy, ch) * 0.16;
            }
        }
        let center = sample_raw(i32(px), i32(py), ch);
        convolved[ch * plane + out_base] = sum - center * 4.0;
    }
}
