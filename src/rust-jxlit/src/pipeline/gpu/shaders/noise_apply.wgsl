struct NoiseApplyParams {
    frame_width: u32,
    frame_height: u32,
    region_left: u32,
    region_top: u32,
    region_width: u32,
    region_height: u32,
    grid_stride: u32,
    grid_off_x: u32,
    grid_off_y: u32,
    corr_x: f32,
    corr_b: f32,
}

struct LutUniform {
    rows: array<vec4<f32>, 3>,
}

@group(0) @binding(0) var<uniform> params: NoiseApplyParams;
@group(0) @binding(1) var<uniform> lut_uniform: LutUniform;
@group(0) @binding(2) var<storage, read> convolved: array<f32>;
@group(0) @binding(3) var<storage, read_write> ch_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> ch_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> ch_b: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fx = gid.x;
    let fy = gid.y;
    if fx >= params.region_width || fy >= params.region_height {
        return;
    }

    let x = fx + params.region_left;
    let y = fy + params.region_top;
    let plane = params.frame_width * params.frame_height;
    let noise_idx = y * params.frame_width + x;
    let grid_idx = (fy + params.grid_off_y) * params.grid_stride + (fx + params.grid_off_x);

    let grid_x = ch_x[grid_idx];
    let grid_y = ch_y[grid_idx];
    let noise_x = convolved[noise_idx];
    let noise_y = convolved[plane + noise_idx];
    let noise_b = convolved[2u * plane + noise_idx];

    let lut_rows = lut_uniform.rows;
    let lut = array<f32, 9>(
        lut_rows[0].x,
        lut_rows[0].y,
        lut_rows[0].z,
        lut_rows[0].w,
        lut_rows[1].x,
        lut_rows[1].y,
        lut_rows[1].z,
        lut_rows[1].w,
        lut_rows[2].x,
    );

    let in_x = grid_x + grid_y;
    let in_y = grid_y - grid_x;
    let in_scaled_x = max(0.0, in_x * 3.0);
    let in_scaled_y = max(0.0, in_y * 3.0);

    let in_x_int = min(u32(in_scaled_x), 7u);
    let in_x_frac = in_scaled_x - f32(in_x_int);
    let in_y_int = min(u32(in_scaled_y), 7u);
    let in_y_frac = in_scaled_y - f32(in_y_int);

    let sx = (lut[in_x_int + 1u] - lut[in_x_int]) * in_x_frac + lut[in_x_int];
    let sy = (lut[in_y_int + 1u] - lut[in_y_int]) * in_y_frac + lut[in_y_int];
    let nx = 0.22 * sx * (0.0078125 * noise_x + 0.9921875 * noise_b);
    let ny = 0.22 * sy * (0.0078125 * noise_y + 0.9921875 * noise_b);

    ch_x[grid_idx] += params.corr_x * (nx + ny) + nx - ny;
    ch_y[grid_idx] += nx + ny;
    ch_b[grid_idx] += params.corr_b * (nx + ny);
}
