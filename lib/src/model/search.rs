use crate::BoxError;
use reqwest::Client;
use serde_json::Value;

/// Search for repositories by keyword(s) and optional filter; returns the
/// matching `modelId`s.
pub async fn search(
    keywords: &[String],
    filter: Option<&str>,
    api_base_url: &str,
) -> Result<Vec<String>, BoxError> {
    let client = Client::new();
    let url = build_search_url(keywords, filter, api_base_url);
    let response = client.get(&url).send().await?;
    if response.status().is_success() {
        let models: Value = response.json().await?;
        Ok(model_ids(&models))
    } else {
        let query = keywords.join(" ");
        Err(format!(
            "Failed to search models with keyword '{query}' (HTTP {})",
            response.status()
        )
        .into())
    }
}

/// Extract `modelId`s from a search response array.
pub fn model_ids(models: &Value) -> Vec<String> {
    models
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|m| {
                    m.get("modelId")
                        .and_then(|id| id.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn build_search_url(keywords: &[String], filter: Option<&str>, api_base_url: &str) -> String {
    let search_query = keywords.join(" ");
    let mut url = format!("{api_base_url}/api/models?search={search_query}");
    if let Some(f) = filter {
        url.push_str(&format!("&filter={f}"));
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

    #[test]
    fn test_model_ids() {
        let json: Value =
            serde_json::from_str(r#"[{"modelId": "a/b"}, {"modelId": "c/d"}, {"other": "x"}]"#)
                .unwrap();
        assert_eq!(model_ids(&json), vec!["a/b", "c/d"]);
    }
}
