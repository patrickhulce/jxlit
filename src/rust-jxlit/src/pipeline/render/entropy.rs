//! Scaffolding placeholder: entropy decoding of frame groups.
//!
//! In the current pipeline this happens inside
//! `jxl_render::RenderContext::render_keyframe` (via the private
//! `jxl-render`/`jxl-frame` group decode path, e.g.
//! `jxl_frame::data::decode_pass_group`). It will move here once the renderer is
//! forked to consume the per-tile `FramePlan` directly.
