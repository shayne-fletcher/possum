use reqwest::Client;
use serde_json::Value;
use std::error::Error;

fn extract_branch_names(parsed_json: &Value) -> Result<(), Box<dyn Error + Send + Sync>> {
    if let Some(branches) = parsed_json.get("branches").and_then(|b| b.as_array()) {
        for branch in branches {
            if let Some(name) = branch.get("name").and_then(|n| n.as_str()) {
                println!("{}", name);
            }
        }
    } else {
        println!("No branches found.");
    }

    Ok(())
}

/// Function to discover revisions (branches or tags) for a Hugging
/// Face repository
pub async fn revisions(repository: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = format!("https://huggingface.co/api/models/{}/refs", repository);

    let response = client.get(&url).send().await?;

    if response.status().is_success() {
        let refs: Value = response.json().await?;
        extract_branch_names(&refs)?;

        Ok(())
    } else {
        println!("Failed to fetch refs: {}", response.status());
        Err(format!("Failed to fetch refs for '{}'", repository).into())
    }
}
