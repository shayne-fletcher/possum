use reqwest::Client;
use serde_json::Value;
use std::error::Error;

pub async fn metadata(
    repository: &str,
    api_base_url: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!("{}/api/models/{}", api_base_url, repository);
    let response = client.get(&url).send().await?;
    if response.status().is_success() {
        let metadata: Value = response.json().await?;
        println!("{}", metadata.to_string());
        Ok(())
    } else {
        eprintln!("Failed to fetch metadata: {}", response.status());
        Err(format!("Failed to get metadata for {}", repository).into())
    }
}

pub fn build_metadata_url(repository: &str, api_base_url: &str) -> String {
    format!("{}/api/models/{}", api_base_url, repository)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_metadata_url() {
        let url = build_metadata_url("TheBloke/Llama-2-7B-Chat-GPTQ", "https://huggingface.co");
        assert_eq!(
            url,
            "https://huggingface.co/api/models/TheBloke/Llama-2-7B-Chat-GPTQ"
        );
    }

    #[test]
    fn test_build_metadata_url_simple_name() {
        let url = build_metadata_url("gpt2", "https://huggingface.co");
        assert_eq!(url, "https://huggingface.co/api/models/gpt2");
    }

    #[test]
    fn test_build_metadata_url_with_special_chars() {
        let url = build_metadata_url("microsoft/DialoGPT-medium", "https://huggingface.co");
        assert_eq!(
            url,
            "https://huggingface.co/api/models/microsoft/DialoGPT-medium"
        );
    }

    #[test]
    fn test_build_metadata_url_custom_base() {
        let url = build_metadata_url("test/model", "http://localhost:8080");
        assert_eq!(url, "http://localhost:8080/api/models/test/model");
    }
}
