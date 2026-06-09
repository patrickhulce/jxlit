//! Scaffolding placeholder: inverse DCT (frequency -> spatial domain).
//!
//! In the current pipeline this happens inside
//! `jxl_render::RenderContext::render_keyframe` (private `jxl-render` `vardct`
//! transforms). It will move here once the renderer is forked to run the
//! transform per-tile (a prime GPU candidate).
