//! Render scope orchestrators and per-encoding flow implementations.
//!
//! `region` / `tile` / `frame` orchestrate at their respective scopes;
//! `flows::{vardct,modular}` implement the per-frame coefficient -> pixel path
//! that `render_frame` dispatches to.

pub mod flows;
pub mod frame;
pub mod region;
pub mod tile;
