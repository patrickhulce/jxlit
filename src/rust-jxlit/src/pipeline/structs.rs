//! Pipeline data structures, split along a Declaration vs Ctx boundary.
//!
//! A `*Declaration` is the cheap, immutable description / identity of a thing;
//! a `*Ctx` is the rich, materialized context (buffers / caches / decoded
//! tables) needed to realize it.

pub mod container;
pub mod frame;
pub mod tile;
