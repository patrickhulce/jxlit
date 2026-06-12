struct GridParams {
    width: u32,
    height: u32,
}

struct XybParams {
    width: u32,
    height: u32,
    opsin_bias_x: f32,
    opsin_bias_y: f32,
    opsin_bias_z: f32,
    intensity_target: f32,
}

struct MatrixParams {
    width: u32,
    height: u32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
}

struct LumaXyzParams {
    width: u32,
    height: u32,
    illuminant_x: f32,
    illuminant_y: f32,
}

struct TransferParams {
    width: u32,
    height: u32,
    tf_kind: u32,
    inverse: u32,
    gamma: f32,
    intensity_target: f32,
    luminance_r: f32,
    luminance_g: f32,
    luminance_b: f32,
}

struct GamutParams {
    width: u32,
    height: u32,
    luminance_r: f32,
    luminance_g: f32,
    luminance_b: f32,
    saturation_factor: f32,
}

struct ToneMapParams {
    width: u32,
    height: u32,
    luminance_r: f32,
    luminance_g: f32,
    luminance_b: f32,
    intensity_target: f32,
    min_nits: f32,
    target_display_luminance: f32,
    peak_luminance: f32,
}

struct ToneMapLumaParams {
    width: u32,
    height: u32,
    intensity_target: f32,
    min_nits: f32,
    target_display_luminance: f32,
    peak_luminance: f32,
}

fn idx_at(p: GridParams, x: u32, y: u32) -> u32 {
    return x + y * p.width;
}

fn apply_gamma(v: f32, gamma: f32) -> f32 {
    if abs(v) <= 1e-7 {
        return 0.0;
    }
    return sign(v) * pow(abs(v), gamma);
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn linear_to_bt709(v: f32) -> f32 {
    if v < 0.018 {
        return 4.5 * v;
    }
    return 1.099 * pow(v, 0.45) - 0.099;
}

fn bt709_to_linear(v: f32) -> f32 {
    if v < 0.081 {
        return v / 4.5;
    }
    return pow((v + 0.099) / 1.099, 1.0 / 0.45);
}

fn linear_to_pq(v: f32, intensity_target: f32) -> f32 {
    let y = v / intensity_target;
    let y_p = pow(y, 0.1593017578125);
    let num = 0.8359375 + 18.8515625 * y_p;
    let den = 1.0 + 18.6875 * y_p;
    return pow(num / den, 78.84375);
}

fn pq_to_linear(v: f32, intensity_target: f32) -> f32 {
    let x_p = pow(v, 1.0 / 78.84375);
    let num = max(x_p - 0.8359375, 0.0);
    let den = 18.8525625 - 18.6875 * x_p;
    return pow(num / den, 1.0 / 0.1593017578125) * intensity_target;
}

fn linear_to_hlg(v: f32) -> f32 {
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.5599107;
    if v <= 1.0 / 12.0 {
        return sqrt(3.0 * v);
    }
    return a * log(12.0 * v - b) + c;
}

fn hlg_to_linear(v: f32) -> f32 {
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.5599107;
    if v <= 0.5 {
        return v * v / 3.0;
    }
    return (exp((v - c) / a) + b) / 12.0;
}

fn rec2408_eetf(from_pq: f32, intensity_target: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    var lb = from_min / intensity_target;
    var lw = from_max / intensity_target;
    var lmin = to_min / intensity_target;
    var lmax = to_max / intensity_target;
    lb = linear_to_pq(lb * intensity_target, intensity_target);
    lw = linear_to_pq(lw * intensity_target, intensity_target);
    lmin = linear_to_pq(lmin * intensity_target, intensity_target);
    lmax = linear_to_pq(lmax * intensity_target, intensity_target);

    let source_pq_diff = lw - lb;
    let normalized = (from_pq - lb) / source_pq_diff;
    let min_luminance = (lmin - lb) / source_pq_diff;
    let max_luminance = (lmax - lb) / source_pq_diff;
    let ks = 1.5 * max_luminance - 0.5;
    let b = min_luminance;

    var compressed = normalized;
    if normalized >= ks {
        let one_sub_ks = 1.0 - ks;
        let t = (normalized - ks) / one_sub_ks;
        let t_p2 = t * t;
        let t_p3 = t_p2 * t;
        compressed = (2.0 * t_p3 - 3.0 * t_p2 + 1.0) * ks
            + (t_p3 - 2.0 * t_p2 + t) * one_sub_ks
            + (-2.0 * t_p3 + 3.0 * t_p2) * max_luminance;
    }
    let one_sub_c = 1.0 - compressed;
    let normalized_target = one_sub_c * one_sub_c * one_sub_c * one_sub_c * b + compressed;
    return normalized_target * source_pq_diff + lb;
}

fn map_gamut(rgb: vec3<f32>, luminance: vec3<f32>, saturation_factor: f32) -> vec3<f32> {
    let y = dot(rgb, luminance);
    var gray_saturation = 0.0;
    var gray_luminance = 0.0;
    for (var i = 0; i < 3; i = i + 1) {
        let v = rgb[i];
        let v_sub_y = v - y;
        let inv = select(v_sub_y, 1.0, v_sub_y == 0.0);
        let v_over = v / inv;
        if v_sub_y < 0.0 {
            gray_saturation = max(gray_saturation, v_over);
        }
        if v_sub_y > 0.0 {
            gray_luminance = max(gray_luminance, v_over - inv);
        }
    }
    let gray_mix = clamp(saturation_factor * (gray_saturation - gray_luminance) + gray_luminance, 0.0, 1.0);
    var mixed = vec3<f32>(
        gray_mix * (y - rgb.x) + rgb.x,
        gray_mix * (y - rgb.y) + rgb.y,
        gray_mix * (y - rgb.z) + rgb.z,
    );
    let max_color = max(max(mixed.x, mixed.y), mixed.z);
    let denom = max(max_color, 1.0);
    return mixed / denom;
}

fn apply_transfer_sample(v: f32, p: TransferParams) -> f32 {
    if p.tf_kind == 0u { return v; } // Linear
    if p.tf_kind == 1u {
        if p.inverse != 0u { return srgb_to_linear(v); }
        return linear_to_srgb(v);
    }
    if p.tf_kind == 2u {
        if p.inverse != 0u { return bt709_to_linear(v); }
        return linear_to_bt709(v);
    }
    if p.tf_kind == 3u {
        let g = select(1e7 / p.gamma, p.gamma / 1e7, p.inverse != 0u);
        return apply_gamma(v, g);
    }
    if p.tf_kind == 4u {
        if p.inverse != 0u { return pq_to_linear(v, p.intensity_target); }
        return linear_to_pq(v, p.intensity_target);
    }
    if p.tf_kind == 5u {
        return apply_gamma(v, select(1.0 / 2.6, 2.6, p.inverse != 0u));
    }
    return v;
}

@group(0) @binding(0) var<uniform> xyb_params: XybParams;
@group(0) @binding(1) var<storage, read_write> ch0: array<f32>;
@group(0) @binding(2) var<storage, read_write> ch1: array<f32>;
@group(0) @binding(3) var<storage, read_write> ch2: array<f32>;

fn cbrt_signed(v: f32) -> f32 {
    return sign(v) * pow(abs(v), 1.0 / 3.0);
}

@compute @workgroup_size(8, 8, 1)
fn xyb_to_lms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= xyb_params.width || y >= xyb_params.height { return; }
    let i = idx_at(GridParams(xyb_params.width, xyb_params.height), x, y);
    let xv = ch0[i];
    let yv = ch1[i];
    let bv = ch2[i];
    let itscale = 255.0 / xyb_params.intensity_target;
    let cbrt_ob = vec3<f32>(
        cbrt_signed(xyb_params.opsin_bias_x),
        cbrt_signed(xyb_params.opsin_bias_y),
        cbrt_signed(xyb_params.opsin_bias_z),
    );
    let ob = vec3<f32>(xyb_params.opsin_bias_x, xyb_params.opsin_bias_y, xyb_params.opsin_bias_z);
    let g_l = yv + xv - cbrt_ob.x;
    let g_m = yv - xv - cbrt_ob.y;
    let g_s = bv - cbrt_ob.z;
    ch0[i] = (g_l * g_l * g_l + ob.x) * itscale;
    ch1[i] = (g_m * g_m * g_m + ob.y) * itscale;
    ch2[i] = (g_s * g_s * g_s + ob.z) * itscale;
}

@group(0) @binding(0) var<uniform> matrix_params: MatrixParams;

@compute @workgroup_size(8, 8, 1)
fn matrix3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= matrix_params.width || y >= matrix_params.height { return; }
    let i = idx_at(GridParams(matrix_params.width, matrix_params.height), x, y);
    let v = vec3(ch0[i], ch1[i], ch2[i]);
    let mat = mat3x3<f32>(
        vec3(matrix_params.m00, matrix_params.m10, matrix_params.m20),
        vec3(matrix_params.m01, matrix_params.m11, matrix_params.m21),
        vec3(matrix_params.m02, matrix_params.m12, matrix_params.m22),
    );
    let r = mat * v;
    ch0[i] = r.x;
    ch1[i] = r.y;
    ch2[i] = r.z;
}

@group(0) @binding(0) var<uniform> luma_xyz_params: LumaXyzParams;

@compute @workgroup_size(8, 8, 1)
fn luma_to_xyz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= luma_xyz_params.width || y >= luma_xyz_params.height { return; }
    let i = idx_at(GridParams(luma_xyz_params.width, luma_xyz_params.height), x, y);
    let a = ch0[i];
    let ix = luma_xyz_params.illuminant_x;
    let iy = luma_xyz_params.illuminant_y;
    let luma_div_y = a / iy;
    ch0[i] = ix * luma_div_y;
    ch1[i] = a;
    ch2[i] = (1.0 - ix - iy) * luma_div_y;
}

@compute @workgroup_size(8, 8, 1)
fn xyz_to_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= luma_xyz_params.width || y >= luma_xyz_params.height { return; }
    let i = idx_at(GridParams(luma_xyz_params.width, luma_xyz_params.height), x, y);
    ch0[i] = ch1[i];
}

@group(0) @binding(0) var<uniform> transfer_params: TransferParams;

@compute @workgroup_size(8, 8, 1)
fn transfer_fn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= transfer_params.width || y >= transfer_params.height { return; }
    let i = idx_at(GridParams(transfer_params.width, transfer_params.height), x, y);
    if transfer_params.tf_kind == 6u {
        let lr = transfer_params.luminance_r;
        let lg = transfer_params.luminance_g;
        let lb = transfer_params.luminance_b;
        let it = transfer_params.intensity_target;
        if (it >= 295.0) && (it <= 305.0) { return; }
        let gamma = 1.2 * pow(1.111, log2(it / 1000.0));
        let exp = (1.0 - gamma) / gamma;
        var rv = ch0[i]; var gv = ch1[i]; var bv = ch2[i];
        if transfer_params.inverse == 0u {
            let mixed = rv * lr + gv * lg + bv * lb;
            let mult = pow(mixed, exp);
            rv *= mult; gv *= mult; bv *= mult;
            ch0[i] = linear_to_hlg(rv);
            ch1[i] = linear_to_hlg(gv);
            ch2[i] = linear_to_hlg(bv);
        } else {
            rv = hlg_to_linear(rv);
            gv = hlg_to_linear(gv);
            bv = hlg_to_linear(bv);
            let mixed = rv * lr + gv * lg + bv * lb;
            let mult = pow(mixed, gamma - 1.0);
            ch0[i] = rv * mult;
            ch1[i] = gv * mult;
            ch2[i] = bv * mult;
        }
        return;
    }
    ch0[i] = apply_transfer_sample(ch0[i], transfer_params);
    ch1[i] = apply_transfer_sample(ch1[i], transfer_params);
    ch2[i] = apply_transfer_sample(ch2[i], transfer_params);
}

@group(0) @binding(0) var<uniform> hlg_params: TransferParams;

@compute @workgroup_size(8, 8, 1)
fn hlg_inverse_ootf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= hlg_params.width || y >= hlg_params.height { return; }
    let i = idx_at(GridParams(hlg_params.width, hlg_params.height), x, y);
    let it = hlg_params.intensity_target;
    if (it >= 295.0) && (it <= 305.0) { return; }
    let lr = hlg_params.luminance_r;
    let lg = hlg_params.luminance_g;
    let lb = hlg_params.luminance_b;
    let gamma = 1.2 * pow(1.111, log2(it / 1000.0));
    let exp = (1.0 - gamma) / gamma;
    var rv = ch0[i]; var gv = ch1[i]; var bv = ch2[i];
    let mixed = rv * lr + gv * lg + bv * lb;
    let mult = pow(mixed, exp);
    ch0[i] = rv * mult;
    ch1[i] = gv * mult;
    ch2[i] = bv * mult;
}

@group(0) @binding(0) var<uniform> gamut_params: GamutParams;

@compute @workgroup_size(8, 8, 1)
fn gamut_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= gamut_params.width || y >= gamut_params.height { return; }
    let i = idx_at(GridParams(gamut_params.width, gamut_params.height), x, y);
    let rgb = map_gamut(vec3(ch0[i], ch1[i], ch2[i]), vec3(gamut_params.luminance_r, gamut_params.luminance_g, gamut_params.luminance_b), gamut_params.saturation_factor);
    ch0[i] = rgb.x;
    ch1[i] = rgb.y;
    ch2[i] = rgb.z;
}

@group(0) @binding(0) var<uniform> clip_params: GridParams;

@compute @workgroup_size(8, 8, 1)
fn clip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= clip_params.width || y >= clip_params.height { return; }
    let i = idx_at(clip_params, x, y);
    ch0[i] = clamp(ch0[i], 0.0, 1.0);
    ch1[i] = clamp(ch1[i], 0.0, 1.0);
    ch2[i] = clamp(ch2[i], 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn invert_channels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= clip_params.width || y >= clip_params.height { return; }
    let i = idx_at(clip_params, x, y);
    ch0[i] = 1.0 - ch0[i];
    ch1[i] = 1.0 - ch1[i];
    ch2[i] = 1.0 - ch2[i];
}

@compute @workgroup_size(8, 8, 1)
fn ycbcr_to_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= clip_params.width || y >= clip_params.height { return; }
    let i = idx_at(clip_params, x, y);
    let cb = ch0[i];
    let yy = ch1[i] + 128.0 / 255.0;
    let cr = ch2[i];
    ch0[i] = cr * 1.402 + yy;
    ch1[i] = cb * (-0.114 * 1.772 / 0.587) + cr * (-0.299 * 1.402 / 0.587) + yy;
    ch2[i] = cb * 1.772 + yy;
}

@group(0) @binding(0) var<uniform> tone_params: ToneMapParams;

@compute @workgroup_size(8, 8, 1)
fn tone_map_rgb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= tone_params.width || y >= tone_params.height { return; }
    let i = idx_at(GridParams(tone_params.width, tone_params.height), x, y);
    let lr = tone_params.luminance_r;
    let lg = tone_params.luminance_g;
    let lb = tone_params.luminance_b;
    var rv = ch0[i]; var gv = ch1[i]; var bv = ch2[i];
    let lum = rv * lr + gv * lg + bv * lb;
    let from_max = min(tone_params.intensity_target, tone_params.peak_luminance);
    let scale = tone_params.intensity_target / tone_params.target_display_luminance;
    let y_pq = linear_to_pq(lum, tone_params.intensity_target);
    let y_mapped_pq = rec2408_eetf(y_pq, tone_params.intensity_target, tone_params.min_nits, from_max, 0.0, tone_params.target_display_luminance);
    let y_mapped = pq_to_linear(y_mapped_pq, tone_params.intensity_target);
    let ratio = select(y_mapped * scale, y_mapped / lum * scale, abs(lum) > 1e-7);
    ch0[i] = rv * ratio;
    ch1[i] = gv * ratio;
    ch2[i] = bv * ratio;
}

@group(0) @binding(0) var<uniform> tone_luma_params: ToneMapLumaParams;

@compute @workgroup_size(8, 8, 1)
fn tone_map_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= tone_luma_params.width || y >= tone_luma_params.height { return; }
    let i = idx_at(GridParams(tone_luma_params.width, tone_luma_params.height), x, y);
    let from_max = min(tone_luma_params.intensity_target, tone_luma_params.peak_luminance);
    let scale = tone_luma_params.intensity_target / tone_luma_params.target_display_luminance;
    let lum = ch0[i];
    let y_pq = linear_to_pq(lum, tone_luma_params.intensity_target);
    let y_mapped_pq = rec2408_eetf(y_pq, tone_luma_params.intensity_target, tone_luma_params.min_nits, from_max, 0.0, tone_luma_params.target_display_luminance);
    let y_mapped = pq_to_linear(y_mapped_pq, tone_luma_params.intensity_target);
    ch0[i] = y_mapped * scale;
}