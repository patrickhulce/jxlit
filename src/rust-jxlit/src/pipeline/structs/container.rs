//! Container/image-scope structures.

use std::sync::Arc;

use jxl_image::ImageHeader;

use crate::vendor::jxl_render::RenderContext;

/// Cheap, immutable description of the parsed container: the image header, the
/// decoded embedded ICC profile (if any), and the byte offset of the first real
/// frame within the codestream.
pub struct ContainerDeclaration {
    pub image_header: Arc<ImageHeader>,
    pub embedded_icc: Option<Vec<u8>>,
    pub offset: usize,
}

/// Rich container context: the declaration plus the vendored `RenderContext`
/// (loaded frames, blend and color-transform machinery) that frames are decoded
/// through.
pub struct ContainerCtx {
    pub declaration: ContainerDeclaration,
    pub render_context: RenderContext,
}
