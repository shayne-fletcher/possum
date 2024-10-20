use reqwest::Client;
use serde_json::Value;
use std::error::Error;

pub async fn search(
    keywords: &Vec<String>,
    filter: Option<&str>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();

    let search_query = keywords.join(" ");
    let mut url = format!("https://huggingface.co/api/models?search={}", search_query);

    // If a filter (e.g., task type or another keyword like 'gptq') is provided, add it to the query
    if let Some(f) = filter {
        url.push_str(&format!("&filter={}", f));
    }

    let response = client.get(&url).send().await?;

    if response.status().is_success() {
        let models: Value = response.json().await?;

        // Check if we received an array of models
        if let Some(model_list) = models.as_array() {
            // Print the modelId of each model
            for model in model_list {
                if let Some(model_id) = model.get("modelId").and_then(|id| id.as_str()) {
                    println!("{}", model_id);
                }
            }
        } else {
            println!("No models found for '{}'.", search_query);
        }

        Ok(())
    } else {
        println!("Failed to search models: {}", response.status());
        Err(format!("Failed to search models with keyword '{}'", search_query).into())
    }
}
