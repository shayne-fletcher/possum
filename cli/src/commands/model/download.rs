use futures::future::join_all;
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::sync::Arc;
use tokio::task;

pub async fn list_files(
    repository: &String,
    revision: Option<&String>,
    token: Option<&String>,
    api_base_url: &str,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let url = build_file_list_url(repository, revision.map(|s| s.as_str()), api_base_url);

    tracing::info!(
        "Getting a file list of {repository} (@ revision \"{}\")",
        revision.unwrap_or(&"main".to_owned())
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

pub async fn download(
    repository: &String,
    revision: Option<&String>,
    to: &std::path::PathBuf,
    token: Option<&String>,
    api_base_url: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if !to.exists() {
        fs::create_dir_all(to)?;
        tracing::info!("Created directory: {}", to.display());
    }

    let files = list_files(repository, revision, token, api_base_url).await?;

    let has_safetensor = files
        .iter()
        .any(|file| file.starts_with("model") && file.ends_with(".safetensors"));

    let client = Arc::new(Client::new());
    let mp = Arc::new(MultiProgress::new()); // MultiProgress for managing multiple progress bars
    let download_tasks: Vec<_> = files
        .into_iter()
        .filter(|file| !should_ignore_file(file, has_safetensor))
        .map(|file| {
            let client = Arc::clone(&client);
            let token = token.cloned();
            let to = to.clone();
            let repository = repository.clone();
            let revision = revision.cloned();
            let mp = Arc::clone(&mp);
            let api_base_url = api_base_url.to_string();

            task::spawn(async move {
                let url = build_download_url(
                    &repository,
                    revision.as_deref(),
                    &file,
                    &api_base_url,
                );

                let request = match token {
                    Some(t) => client.get(&url).bearer_auth(t),
                    None => client.get(&url),
                };

                let response = request.send().await?;
                if response.status().is_success() {
                    let file_path = to.join(&file);
                    let total_size = response.content_length().unwrap_or(0);

                    let progress_bar = mp.add(ProgressBar::new(total_size));
                    progress_bar.set_style(
                        ProgressStyle::default_bar()
                            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta}) - {msg}")
                            .expect("Failed to create ProgressBar template")
                            .progress_chars("#>-"),
                    );
                    progress_bar.set_message(file.clone());

                    let mut dest = tokio::fs::File::create(&file_path).await?;
                    let mut content = response.bytes_stream();
                    while let Some(chunk) = content.next().await {
                        let chunk = chunk?;
                        tokio::io::copy(&mut chunk.as_ref(), &mut dest).await?;
                        progress_bar.inc(chunk.len() as u64);
                    }
                    progress_bar.finish_with_message(format!("Downloaded: {file}"));
                } else {
                    tracing::error!("Failed to download file: {}", file);
                }

                Ok::<(), Box<dyn Error + Send + Sync>>(())
            })
        })
        .collect();

    tracing::info!(
        "Downloading files from {} (@ revision \"{}\")",
        repository,
        revision.unwrap_or(&"main".to_owned())
    );

    join_all(download_tasks).await;

    tracing::info!(
        "Finished downloading from {} (@ revision \"{}\")",
        repository,
        revision.unwrap_or(&"main".to_owned())
    );

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
}
