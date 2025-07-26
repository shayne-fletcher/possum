use reqwest::Client;
use serde_json::Value;
use std::error::Error;

pub async fn search(
    keywords: &Vec<String>,
    filter: Option<&str>,
    api_base_url: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();

    let search_query = keywords.join(" ");
    let mut url = format!("{}/api/models?search={}", api_base_url, search_query);

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

pub fn build_search_url(keywords: &[String], filter: Option<&str>, api_base_url: &str) -> String {
    let search_query = keywords.join(" ");
    let mut url = format!("{}/api/models?search={}", api_base_url, search_query);

    if let Some(f) = filter {
        url.push_str(&format!("&filter={}", f));
    }

    url
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_search_url_with_keywords_only() {
        let keywords = vec!["TheBloke".to_string(), "Llama-2-7B".to_string()];
        let url = build_search_url(&keywords, None, "https://huggingface.co");

        assert_eq!(
            url,
            "https://huggingface.co/api/models?search=TheBloke Llama-2-7B"
        );
    }

    #[test]
    fn test_build_search_url_with_filter() {
        let keywords = vec!["TheBloke".to_string(), "Llama-2-7B".to_string()];
        let url = build_search_url(&keywords, Some("gptq"), "https://huggingface.co");

        assert_eq!(
            url,
            "https://huggingface.co/api/models?search=TheBloke Llama-2-7B&filter=gptq"
        );
    }

    #[test]
    fn test_build_search_url_single_keyword() {
        let keywords = vec!["llama".to_string()];
        let url = build_search_url(&keywords, None, "https://huggingface.co");

        assert_eq!(url, "https://huggingface.co/api/models?search=llama");
    }

    #[test]
    fn test_build_search_url_empty_keywords() {
        let keywords: Vec<String> = vec![];
        let url = build_search_url(
            &keywords,
            Some("text-classification"),
            "https://huggingface.co",
        );

        assert_eq!(
            url,
            "https://huggingface.co/api/models?search=&filter=text-classification"
        );
    }

    #[test]
    fn test_build_search_url_custom_base() {
        let keywords = vec!["test".to_string()];
        let url = build_search_url(&keywords, None, "http://localhost:8080");

        assert_eq!(url, "http://localhost:8080/api/models?search=test");
    }
}
