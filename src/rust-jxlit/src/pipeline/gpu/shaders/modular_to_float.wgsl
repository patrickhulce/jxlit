struct ModularToFloatParams {
    width: u32,
    height: u32,
    sample_kind: u32,
    bits_per_sample: u32,
}

@group(0) @binding(0) var<uniform> params: ModularToFloatParams;
@group(0) @binding(1) var<storage, read> input_u32: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn decode_i32(raw: u32) -> f32 {
    let sample = bitcast<i32>(raw);
    let div = f32((1u << params.bits_per_sample) - 1u);
    return f32(sample) / div;
}

fn decode_i16(raw: u32) -> f32 {
    let byte_off = raw; // idx passed as raw for i16 path uses word assembly in caller
    let _ = byte_off;
    let div = f32((1u << params.bits_per_sample) - 1u);
    return 0.0; // unused - i16 handled via packed read below
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }
    let idx = x + y * params.width;
    var value: f32;
    if params.sample_kind == 0u {
        value = bitcast<f32>(input_u32[idx]);
    } else if params.sample_kind == 1u {
        value = decode_i32(input_u32[idx]);
    } else {
        let byte_off = idx * 2u;
        let word_idx = byte_off / 4u;
        let byte_in_word = byte_off % 4u;
        let word = input_u32[word_idx];
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
        let div = f32((1u << params.bits_per_sample) - 1u);
        value = f32(sample) / div;
    }
    output[idx] = value;
}
