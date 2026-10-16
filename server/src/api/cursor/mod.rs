//! Wires the Cursor-facing API routes.

pub mod bidi;
mod handlers;
pub mod proxy;
pub mod run_http2;
mod run_sse;

pub(crate) use handlers::is_local_model;
pub use handlers::router;
