use crate::BoxError;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

/// Whether [`download`] renders progress bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressMode {
    /// Render `indicatif` progress bars (the default).
    #[default]
    Auto,
    /// No progress UI (e.g. when an embedding caller wants silence).
    Hidden,
}

/// A request to download the files of a repository revision.
///
/// Construct with `Default` + struct-update, e.g.
/// `DownloadRequest { repository, to, include, ..Default::default() }`.
#[derive(Debug, Clone)]
pub struct DownloadRequest {
    /// The model repository id (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`).
    pub repository: String,
    /// Optional revision (branch/tag); `None` means `main`.
    pub revision: Option<String>,
    /// Destination leaf directory; files are written directly under it.
    pub to: PathBuf,
    /// Hugging Face token (for gated repositories).
    pub token: Option<String>,
    /// Glob(s) of files to include (empty ⇒ all).
    pub include: Vec<String>,
    /// Glob(s) of files to exclude.
    pub exclude: Vec<String>,
    /// Maximum number of concurrent file downloads.
    pub concurrency: usize,
    /// Base URL for the Hugging Face API.
    pub api_base_url: String,
    /// Whether to render progress bars.
    pub progress: ProgressMode,
}

impl Default for DownloadRequest {
    fn default() -> Self {
        Self {
            repository: String::new(),
            revision: None,
            to: PathBuf::new(),
            token: None,
            include: Vec::new(),
            exclude: Vec::new(),
            concurrency: 4,
            api_base_url: "https://huggingface.co".to_string(),
            progress: ProgressMode::Auto,
        }
    }
}

/// List the files of a repository revision (the `siblings` of the model info).
pub async fn list_files(
    repository: &str,
    revision: Option<&str>,
    token: Option<&str>,
    api_base_url: &str,
) -> Result<Vec<String>, BoxError> {
    let url = build_file_list_url(repository, revision, api_base_url);

    tracing::info!(
        "Getting a file list of {repository} (@ revision \"{}\")",
        revision.unwrap_or("main")
    );
    tracing::debug!("File list URL: {url}");

    let client = Client::new();
    let request = match token {
        Some(t) => client.get(&url).bearer_auth(t),
        None => client.get(&url),
    };

    let response = request.send().await?;
    if response.status().is_success() {
        let model_info: Value = response.json().await?;
        if let Some(siblings) = model_info["siblings"].as_array() {
            let files: Vec<String> = siblings
                .iter()
                .filter_map(|f| f["rfilename"].as_str().map(|s| s.to_string()))
                .collect();
            if files.is_empty() {
                tracing::info!("No files found in the repository");
            }
            Ok(files)
        } else {
            tracing::info!("No files found in the repository");
            Ok(vec![])
        }
    } else {
        tracing::error!("Failed to list files: {}", response.status());
        Err(format!("Failed to list files for {repository}").into())
    }
}

/// Download the selected files of a repository revision into `req.to`.
///
/// File selection is driven by [`select_files`]; `req.concurrency` bounds the
/// number of simultaneous downloads and `req.progress` controls the bars.
/// Returns an error if any file fails or the repository cannot be listed.
pub async fn download(req: &DownloadRequest) -> Result<(), BoxError> {
    if !req.to.exists() {
        fs::create_dir_all(&req.to)?;
        tracing::info!("Created directory: {}", req.to.display());
    }

    let files = list_files(
        &req.repository,
        req.revision.as_deref(),
        req.token.as_deref(),
        &req.api_base_url,
    )
    .await?;

    let has_safetensor = files
        .iter()
        .any(|file| file.starts_with("model") && file.ends_with(".safetensors"));

    let files = select_files(files, &req.include, &req.exclude, has_safetensor)?;
    if files.is_empty() {
        tracing::warn!("No files matched the selection; nothing to download");
        return Ok(());
    }

    let concurrency = req.concurrency.max(1);
    let client = Arc::new(Client::new());
    let mp = Arc::new(MultiProgress::new());

    tracing::info!(
        "Downloading {} file(s) from {} (@ revision \"{}\") [concurrency {concurrency}]",
        files.len(),
        req.repository,
        req.revision.as_deref().unwrap_or("main")
    );

    let results: Vec<Result<(), BoxError>> = futures::stream::iter(files.into_iter().map(|file| {
        let client = Arc::clone(&client);
        let mp = Arc::clone(&mp);
        let token = req.token.clone();
        let to = req.to.clone();
        let repository = req.repository.clone();
        let revision = req.revision.clone();
        let api_base_url = req.api_base_url.clone();
        let progress = req.progress;

        async move {
            download_file(
                &client,
                &mp,
                &repository,
                revision.as_deref(),
                &file,
                token.as_deref(),
                &to,
                &api_base_url,
                progress,
            )
            .await
            .inspect_err(|e| tracing::error!("Failed to download {file}: {e}"))
        }
    }))
    .buffer_unordered(concurrency)
    .collect()
    .await;

    let failures = results.iter().filter(|r| r.is_err()).count();

    tracing::info!(
        "Finished downloading from {} (@ revision \"{}\")",
        req.repository,
        req.revision.as_deref().unwrap_or("main")
    );

    if failures > 0 {
        return Err(format!(
            "{failures} file(s) failed to download from {}",
            req.repository
        )
        .into());
    }

    Ok(())
}

// Download a single file: GET it, create any nested parent directories,
// stream it to a temporary `.incomplete` sibling, and rename on success so an
// interrupted download never leaves a truncated file that looks complete.
#[allow(clippy::too_many_arguments)]
async fn download_file(
    client: &Client,
    mp: &MultiProgress,
    repository: &str,
    revision: Option<&str>,
    file: &str,
    token: Option<&str>,
    to: &Path,
    api_base_url: &str,
    progress: ProgressMode,
) -> Result<(), BoxError> {
    let url = build_download_url(repository, revision, file, api_base_url);

    let request = match token {
        Some(t) => client.get(&url).bearer_auth(t),
        None => client.get(&url),
    };

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(format!("{file}: HTTP {}", response.status()).into());
    }

    let file_path = to.join(file);
    if let Some(parent) = file_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let total_size = response.content_length().unwrap_or(0);
    let progress_bar = match progress {
        ProgressMode::Auto => {
            let pb = mp.add(ProgressBar::new(total_size));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta}) - {msg}")
                    .expect("Failed to create ProgressBar template")
                    .progress_chars("#>-"),
            );
            pb.set_message(file.to_string());
            pb
        }
        ProgressMode::Hidden => ProgressBar::hidden(),
    };

    let mut tmp_os = file_path.clone().into_os_string();
    tmp_os.push(".incomplete");
    let tmp_path = PathBuf::from(tmp_os);

    let mut dest = tokio::fs::File::create(&tmp_path).await?;
    let mut content = response.bytes_stream();
    while let Some(chunk) = content.next().await {
        let chunk = chunk?;
        tokio::io::copy(&mut chunk.as_ref(), &mut dest).await?;
        progress_bar.inc(chunk.len() as u64);
    }
    dest.flush().await?;
    drop(dest);
    tokio::fs::rename(&tmp_path, &file_path).await?;

    progress_bar.finish_with_message(format!("Downloaded: {file}"));
    Ok(())
}

pub fn build_file_list_url(repository: &str, revision: Option<&str>, api_base_url: &str) -> String {
    let mut url = format!("{api_base_url}/api/models/{repository}");
    if let Some(rev) = revision {
        url = format!("{url}/revision/{rev}");
    }
    url
}

pub fn build_download_url(
    repository: &str,
    revision: Option<&str>,
    filename: &str,
    api_base_url: &str,
) -> String {
    let revision = revision.unwrap_or("main");
    format!("{api_base_url}/{repository}/resolve/{revision}/{filename}")
}

pub fn should_ignore_file(filename: &str, has_safetensors: bool) -> bool {
    if !has_safetensors {
        return false;
    }

    let ignore_patterns = ["*.pt", "*.bin"];
    ignore_patterns
        .iter()
        .map(|p| glob::Pattern::new(p).unwrap())
        .any(|pattern| pattern.matches(filename))
}

/// Apply file selection: keep files matching any `include` glob (or all when
/// `include` is empty), then drop any matching an `exclude` glob. Only in the
/// default case (no explicit `include`) do we also drop the `*.pt`/`*.bin`
/// duplicates of safetensors weights; an explicit include is taken as the
/// caller knowing exactly what they want.
pub fn select_files(
    files: Vec<String>,
    include: &[String],
    exclude: &[String],
    has_safetensors: bool,
) -> Result<Vec<String>, glob::PatternError> {
    let includes = compile_patterns(include)?;
    let excludes = compile_patterns(exclude)?;

    let selected = files
        .into_iter()
        .filter(|f| includes.is_empty() || includes.iter().any(|p| p.matches(f)))
        .filter(|f| !excludes.iter().any(|p| p.matches(f)))
        .filter(|f| !(includes.is_empty() && should_ignore_file(f, has_safetensors)))
        .collect();

    Ok(selected)
}

fn compile_patterns(patterns: &[String]) -> Result<Vec<glob::Pattern>, glob::PatternError> {
    patterns.iter().map(|p| glob::Pattern::new(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_file_list_url_no_revision() {
        let url = build_file_list_url(
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            None,
            "https://huggingface.co",
        );
        assert_eq!(
            url,
            "https://huggingface.co/api/models/TheBloke/Llama-2-7B-Chat-GPTQ"
        );
    }

    #[test]
    fn test_build_file_list_url_with_revision() {
        let url = build_file_list_url(
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            Some("gptq-4bit-64g-actorder_True"),
            "https://huggingface.co",
        );
        assert_eq!(url, "https://huggingface.co/api/models/TheBloke/Llama-2-7B-Chat-GPTQ/revision/gptq-4bit-64g-actorder_True");
    }

    #[test]
    fn test_build_download_url_main_branch() {
        let url = build_download_url(
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            None,
            "model.safetensors",
            "https://huggingface.co",
        );
        assert_eq!(
            url,
            "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ/resolve/main/model.safetensors"
        );
    }

    #[test]
    fn test_build_download_url_with_revision() {
        let url = build_download_url(
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            Some("gptq-4bit-64g-actorder_True"),
            "model.safetensors",
            "https://huggingface.co",
        );
        assert_eq!(url, "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ/resolve/gptq-4bit-64g-actorder_True/model.safetensors");
    }

    #[test]
    fn test_build_file_list_url_custom_base() {
        let url = build_file_list_url("test/model", None, "http://localhost:8080");
        assert_eq!(url, "http://localhost:8080/api/models/test/model");
    }

    #[test]
    fn test_build_download_url_custom_base() {
        let url = build_download_url("test/model", None, "file.txt", "http://localhost:8080");
        assert_eq!(
            url,
            "http://localhost:8080/test/model/resolve/main/file.txt"
        );
    }

    #[test]
    fn test_should_ignore_file_no_safetensors() {
        assert!(!should_ignore_file("model.bin", false));
        assert!(!should_ignore_file("model.pt", false));
        assert!(!should_ignore_file("model.safetensors", false));
    }

    #[test]
    fn test_should_ignore_file_with_safetensors() {
        assert!(should_ignore_file("model.bin", true));
        assert!(should_ignore_file("model.pt", true));
        assert!(should_ignore_file("pytorch_model.bin", true));
        assert!(!should_ignore_file("model.safetensors", true));
        assert!(!should_ignore_file("config.json", true));
    }

    #[test]
    fn test_should_ignore_file_patterns() {
        assert!(should_ignore_file("anything.pt", true));
        assert!(should_ignore_file("anything.bin", true));
        assert!(!should_ignore_file("anything.safetensors", true));
        assert!(!should_ignore_file("model.json", true));
    }

    #[test]
    fn test_select_files_no_filters_keeps_all() {
        let files = vec!["config.json".to_string(), "model.safetensors".to_string()];
        let out = select_files(files.clone(), &[], &[], false).unwrap();
        assert_eq!(out, files);
    }

    #[test]
    fn test_select_files_include_only() {
        let files = vec![
            "config.json".to_string(),
            "model.safetensors".to_string(),
            "README.md".to_string(),
        ];
        let out = select_files(
            files,
            &["*.safetensors".to_string(), "*.json".to_string()],
            &[],
            true,
        )
        .unwrap();
        assert_eq!(out, vec!["config.json", "model.safetensors"]);
    }

    #[test]
    fn test_select_files_exclude_nested() {
        let files = vec![
            "config.json".to_string(),
            "model.safetensors".to_string(),
            "figures/benchmark.jpg".to_string(),
        ];
        let out = select_files(files, &[], &["figures/*".to_string()], false).unwrap();
        assert_eq!(out, vec!["config.json", "model.safetensors"]);
    }

    #[test]
    fn test_select_files_picks_one_gguf_quant() {
        let files = vec![
            "DeepSeek-Q2_K.gguf".to_string(),
            "DeepSeek-Q4_K_M.gguf".to_string(),
            "DeepSeek-Q8_0.gguf".to_string(),
            "README.md".to_string(),
        ];
        let out = select_files(files, &["*Q4_K_M.gguf".to_string()], &[], false).unwrap();
        assert_eq!(out, vec!["DeepSeek-Q4_K_M.gguf"]);
    }

    #[test]
    fn test_select_files_default_ignores_bin_with_safetensors() {
        let files = vec![
            "model.safetensors".to_string(),
            "pytorch_model.bin".to_string(),
        ];
        let out = select_files(files, &[], &[], true).unwrap();
        assert_eq!(out, vec!["model.safetensors"]);
    }

    #[test]
    fn test_select_files_explicit_include_overrides_bin_heuristic() {
        let files = vec![
            "model.safetensors".to_string(),
            "pytorch_model.bin".to_string(),
        ];
        let out = select_files(files, &["*.bin".to_string()], &[], true).unwrap();
        assert_eq!(out, vec!["pytorch_model.bin"]);
    }

    #[test]
    fn test_select_files_invalid_glob_errors() {
        let files = vec!["a".to_string()];
        assert!(select_files(files, &["[".to_string()], &[], false).is_err());
    }
}
