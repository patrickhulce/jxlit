//! Scaffolding placeholder: XYB -> RGB (output color) transform.
//!
//! In the current pipeline this happens inside
//! `jxl_render::RenderContext::render_keyframe` (private `postprocess_keyframe`
//! using `jxl-color`). It is the prime GPU-acceleration target and will move
//! here once the renderer is forked to expose the pre-transform grids.
