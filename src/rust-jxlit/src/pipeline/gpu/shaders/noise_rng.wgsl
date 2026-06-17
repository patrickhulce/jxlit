struct U64 {
    lo: u32,
    hi: u32,
}

struct NoiseRngParams {
    frame_width: u32,
    frame_height: u32,
    group_dim: u32,
    groups_per_row: u32,
    num_groups: u32,
    visible_frames: u32,
    invisible_frames: u32,
}

struct GroupMeta {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    stride: u32,
    raw_offset: u32,
}

@group(0) @binding(0) var<uniform> params: NoiseRngParams;
@group(0) @binding(1) var<storage, read_write> group_meta: array<GroupMeta>;
@group(0) @binding(2) var<storage, read_write> raw_noise: array<f32>;

const N: u32 = 8u;

fn u64_from(lo: u32, hi: u32) -> U64 {
    return U64(lo, hi);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

fn mul32_to_u64(a: u32, b: u32) -> U64 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + p10 + (p00 >> 16u);
    let lo = (p00 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = p11 + (mid >> 16u);
    return U64(lo, hi);
}

fn mul_u64(a: U64, b: U64) -> U64 {
    let p0 = mul32_to_u64(a.lo, b.lo);
    let p1 = mul32_to_u64(a.lo, b.hi);
    let p2 = mul32_to_u64(a.hi, b.lo);
    let p3 = mul32_to_u64(a.hi, b.hi);
    let mid = u64_add(p1, u64_add(p2, U64(0u, p0.hi)));
    return u64_add(U64(p0.lo, mid.lo), u64_add(U64(0u, mid.hi), p3));
}

fn xor_u64(a: U64, b: U64) -> U64 {
    return U64(a.lo ^ b.lo, a.hi ^ b.hi);
}

fn shr_u64(a: U64, bits: u32) -> U64 {
    if bits >= 32u {
        return U64(0u, a.hi >> (bits - 32u));
    }
    let lo = (a.lo >> bits) | (a.hi << (32u - bits));
    let hi = a.hi >> bits;
    return U64(lo, hi);
}

fn shl_u64(a: U64, bits: u32) -> U64 {
    if bits >= 32u {
        return U64(0u, a.lo << (bits - 32u));
    }
    let lo = a.lo << bits;
    let hi = (a.hi << bits) | (a.lo >> (32u - bits));
    return U64(lo, hi);
}

fn split_mix_64(z: U64) -> U64 {
    var v = xor_u64(z, shr_u64(z, 30u));
    v = mul_u64(v, u64_from(0x1CE4E5B9u, 0xBF58476Du));
    v = xor_u64(v, shr_u64(v, 27u));
    v = mul_u64(v, u64_from(0x133111EBu, 0x94D049BBu));
    v = xor_u64(v, shr_u64(v, 31u));
    return v;
}

fn fill_batch_u32(
    s0: ptr<function, array<U64, 8>>,
    s1: ptr<function, array<U64, 8>>,
) -> array<u32, 16> {
    var out: array<u32, 16>;
    for (var i = 0u; i < N; i = i + 1u) {
        let s1_val = (*s0)[i];
        let s0_val = (*s1)[i];
        let ret = u64_add(s1_val, s0_val);
        (*s0)[i] = s0_val;
        var s1_new = xor_u64(s1_val, shl_u64(s1_val, 23u));
        s1_new = xor_u64(s1_new, xor_u64(s0_val, xor_u64(shr_u64(s1_new, 18u), shr_u64(s0_val, 5u))));
        (*s1)[i] = s1_new;
        out[i * 2u] = ret.lo;
        out[i * 2u + 1u] = ret.hi;
    }
    return out;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let group_idx = gid.x;
    if group_idx >= params.num_groups {
        return;
    }

    let group_x = group_idx % params.groups_per_row;
    let group_y = group_idx / params.groups_per_row;
    let x0 = group_x * params.group_dim;
    let y0 = group_y * params.group_dim;
    let gw = min(params.group_dim, params.frame_width - x0);
    let gh = min(params.group_dim, params.frame_height - y0);
    let width_n2 = (gw + 15u) / 16u;
    let stride = width_n2 * 16u;
    let num_iters = width_n2 * gh;
    let plane_elems = stride * gh;

    let raw_offset = group_idx * 3u * plane_elems;
    group_meta[group_idx] = GroupMeta(x0, y0, gw, gh, stride, raw_offset);

    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        var s0: array<U64, 8>;
        var s1: array<U64, 8>;
        s0[0] = split_mix_64(u64_add(
            u64_from(params.invisible_frames, params.visible_frames),
            u64_from(0x7F4A7C15u, 0x9E3779B9u),
        ));
        s1[0] = split_mix_64(u64_add(
            u64_from(x0, y0),
            u64_from(0x7F4A7C15u, 0x9E3779B9u),
        ));
        for (var i = 1u; i < N; i = i + 1u) {
            s0[i] = split_mix_64(s0[i - 1u]);
            s1[i] = split_mix_64(s1[i - 1u]);
        }

        let ch_base = raw_offset + ch * plane_elems;
        for (var iter = 0u; iter < num_iters; iter = iter + 1u) {
            let batch = fill_batch_u32(&s0, &s1);
            let base = ch_base + iter * 16u;
            for (var j = 0u; j < 16u; j = j + 1u) {
                let u32_val = batch[j];
                raw_noise[base + j] = bitcast<f32>((u32_val >> 9u) | 0x3f800000u);
            }
        }
    }
}
