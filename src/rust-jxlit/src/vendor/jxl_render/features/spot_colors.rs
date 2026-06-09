// Vendored from jxl-oxide (jxl-render 0.12.4), (c) Wonwoo Choi, licensed MIT OR Apache-2.0.
// Source: https://github.com/tirr-c/jxl-oxide/blob/f8ae722ef2d6b782941c89517d19cfbf605c4a9d/crates/jxl-render/src/features/spot_colors.rs
// Copied as-is; only crate-path references changed.

use jxl_grid::AlignedGrid;
use jxl_image::ExtraChannelType;

/// Renders a spot color channel onto color_channels
pub fn render_spot_color(
    mut color_channels: [&mut AlignedGrid<f32>; 3],
    ec_grid: &AlignedGrid<f32>,
    ec_ty: &ExtraChannelType,
) -> crate::vendor::jxl_render::Result<()> {
    let ExtraChannelType::SpotColour {
        red,
        green,
        blue,
        solidity,
    } = ec_ty
    else {
        return Err(crate::vendor::jxl_render::Error::NotSupported(
            "EC type is not SpotColour",
        ));
    };
    if color_channels.len() != 3 {
        return Ok(());
    }

    let spot_colors = [red, green, blue];
    let s = ec_grid.buf();

    (0..3).for_each(|c| {
        let channel = color_channels[c].buf_mut();
        let color = spot_colors[c];
        assert_eq!(channel.len(), s.len());

        (0..channel.len()).for_each(|i| {
            let mix = s[i] * solidity;
            channel[i] = mix * color + (1.0 - mix) * channel[i];
        });
    });
    Ok(())
}
