//! possum — fetch and inspect 🤗 Hugging Face model artifacts as a library.
//!
//! The [`model`] module mirrors the CLI's subcommands as ordinary async
//! functions: [`model::download`], [`model::list_files`], [`model::metadata`],
//! [`model::revisions`], and [`model::search`].

pub mod model;

/// Boxed, thread-safe error used throughout the library.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;
