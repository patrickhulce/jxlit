//! Parse stages: extract bytes from the codestream into our structures.
//!
//! `container` demuxes + parses the image header; `frames` loads frames and the
//! frame-global entropy structures; `tiles` builds the per-tile (pass-group)
//! descriptors and live decode contexts.

pub mod container;
pub mod frames;
pub mod tiles;
