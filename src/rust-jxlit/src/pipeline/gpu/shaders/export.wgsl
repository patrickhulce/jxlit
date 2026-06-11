struct ExportParams {
    orientation: u32,
    out_width: u32,
    out_height: u32,
    channels: u32,
    pixel_layout: u32,
    plane_size: u32,
}

struct ChannelMeta {
    offset_x: i32,
    offset_y: i32,
    grid_width: u32,
    grid_height: u32,
    sample_kind: u32,
    bits_per_sample: u32,
    base_u32: u32,
}

@group(0) @binding(0) var<uniform> params: ExportParams;
@group(0) @binding(1) var<storage, read> channel_meta: array<ChannelMeta, 8>;
@group(0) @binding(2) var<storage, read> channel_data: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn to_original_coord(orientation: u32, width: u32, height: u32, x: u32, y: u32) -> vec2<u32> {
    switch orientation {
        case 1u: { return vec2<u32>(x, y); }
        case 2u: { return vec2<u32>(width - x - 1u, y); }
        case 3u: { return vec2<u32>(width - x - 1u, height - y - 1u); }
        case 4u: { return vec2<u32>(x, height - y - 1u); }
        case 5u: { return vec2<u32>(y, x); }
        case 6u: { return vec2<u32>(y, width - x - 1u); }
        case 7u: { return vec2<u32>(height - y - 1u, width - x - 1u); }
        case 8u: { return vec2<u32>(height - y - 1u, x); }
        default: { return vec2<u32>(x, y); }
    }
}

fn read_u32(base_u32: u32, idx: u32) -> u32 {
    return channel_data[base_u32 + idx];
}

fn decode_sample(base_u32: u32, idx: u32, kind: u32, bits: u32) -> f32 {
    if kind == 0u {
        return bitcast<f32>(read_u32(base_u32, idx));
    }
    if kind == 1u {
        let sample = bitcast<i32>(read_u32(base_u32, idx));
        let div = f32((1u << bits) - 1u);
        return f32(sample) / div;
    }
    let byte_off = idx * 2u;
    let word_idx = base_u32 + byte_off / 4u;
    let byte_in_word = byte_off % 4u;
    let word = channel_data[word_idx];
    var raw: u32;
    if byte_in_word == 0u {
        raw = word & 0xFFFFu;
    } else {
        raw = (word >> 16u) & 0xFFFFu;
    }
    var sample = i32(raw);
    if (raw & 0x8000u) != 0u {
        sample = sample - 65536;
    }
    let div = f32((1u << bits) - 1u);
    return f32(sample) / div;
}

fn sample_at(c: u32, orig_x: u32, orig_y: u32) -> f32 {
    let ch_meta = channel_meta[c];
    let px = i32(orig_x) + ch_meta.offset_x;
    let py = i32(orig_y) + ch_meta.offset_y;
    if px < 0 || py < 0 {
        return 0.0;
    }
    let ux = u32(px);
    let uy = u32(py);
    if ux >= ch_meta.grid_width || uy >= ch_meta.grid_height {
        return 0.0;
    }
    let idx = ux + uy * ch_meta.grid_width;
    return decode_sample(ch_meta.base_u32, idx, ch_meta.sample_kind, ch_meta.bits_per_sample);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.out_width || y >= params.out_height {
        return;
    }

    let orig = to_original_coord(params.orientation, params.out_width, params.out_height, x, y);
    let width = params.out_width;

    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        let value = sample_at(c, orig.x, orig.y);
        var out_idx: u32;
        if params.pixel_layout == 0u {
            out_idx = c + (x + y * width) * params.channels;
        } else {
            out_idx = c * params.plane_size + y * width + x;
        }
        output[out_idx] = value;
    }
}
