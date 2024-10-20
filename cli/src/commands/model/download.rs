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
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut url = format!("https://huggingface.co/api/models/{}", repository);
    if let Some(rev) = revision {
        url = format!("{}/revision/{}", url, rev);
    }

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
        Err(format!("Failed to list files for {}", repository).into())
    }
}

pub async fn download(
    repository: &String,
    revision: Option<&String>,
    to: &std::path::PathBuf,
    token: Option<&String>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if !to.exists() {
        fs::create_dir_all(&to)?;
        tracing::info!("Created directory: {}", to.display());
    }

    let files = list_files(repository, revision, token).await?;

    let has_safetensor = files
        .iter()
        .any(|file| file.starts_with("model") && file.ends_with(".safetensors"));
    let ignore_patterns = if has_safetensor {
        ["*.pt", "*.bin"]
            .iter()
            .map(|p| glob::Pattern::new(p).unwrap())
            .collect()
    } else {
        Vec::new()
    };

    let client = Arc::new(Client::new());
    let mp = Arc::new(MultiProgress::new()); // MultiProgress for managing multiple progress bars
    let download_tasks: Vec<_> = files
        .into_iter()
        .filter(|file| {
            !ignore_patterns.iter().any(|pattern| pattern.matches(file))
        })
        .map(|file| {
            let client = Arc::clone(&client);
            let token = token.cloned();
            let to = to.clone();
            let repository = repository.clone();
            let revision = revision.cloned();
            let mp = Arc::clone(&mp);

            task::spawn(async move {
                let mut url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    repository, file
                );
                if let Some(rev) = revision {
                    url = format!(
                        "https://huggingface.co/{}/resolve/{}/{}",
                        repository, rev, file
                    );
                }

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
                    progress_bar.finish_with_message(format!("Downloaded: {}", file));
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
