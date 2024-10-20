use reqwest::Client;
use serde_json::Value;
use std::error::Error;

pub async fn metadata(repository: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!("https://huggingface.co/api/models/{}", repository);
    let response = client.get(&url).send().await?;
    if response.status().is_success() {
        let metadata: Value = response.json().await?;
        println!("Metadata for {}: {:#?}", repository, metadata);
        Ok(())
    } else {
        eprintln!("Failed to fetch metadata: {}", response.status());
        Err(format!("Failed to get metadata for {}", repository).into())
    }
}
