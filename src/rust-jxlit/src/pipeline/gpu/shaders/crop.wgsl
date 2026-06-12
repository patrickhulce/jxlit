struct CropParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    off_x: u32,
    off_y: u32,
}

@group(0) @binding(0) var<uniform> params: CropParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.dst_width || y >= params.dst_height {
        return;
    }
    let src_idx = (x + params.off_x) + (y + params.off_y) * params.src_width;
    let dst_idx = x + y * params.dst_width;
    output[dst_idx] = input[src_idx];
}
