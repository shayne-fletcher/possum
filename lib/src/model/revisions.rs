use crate::BoxError;
use reqwest::Client;
use serde_json::Value;

/// Discover the revisions (branch names) of a Hugging Face repository.
pub async fn revisions(repository: &str, api_base_url: &str) -> Result<Vec<String>, BoxError> {
    let client = Client::new();
    let url = build_revisions_url(repository, api_base_url);
    let response = client.get(&url).send().await?;
    if response.status().is_success() {
        let refs: Value = response.json().await?;
        Ok(branch_names(&refs))
    } else {
        Err(format!(
            "Failed to fetch refs for '{repository}' (HTTP {})",
            response.status()
        )
        .into())
    }
}

/// Extract branch names from a `/refs` response.
pub fn branch_names(parsed: &Value) -> Vec<String> {
    parsed
        .get("branches")
        .and_then(|b| b.as_array())
        .map(|branches| {
            branches
                .iter()
                .filter_map(|b| b.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

pub fn build_revisions_url(repository: &str, api_base_url: &str) -> String {
    format!("{api_base_url}/api/models/{repository}/refs")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_revisions_url() {
        let url = build_revisions_url("TheBloke/Llama-2-7B-Chat-GPTQ", "https://huggingface.co");
        assert_eq!(
            url,
            "https://huggingface.co/api/models/TheBloke/Llama-2-7B-Chat-GPTQ/refs"
        );
    }

    #[test]
    fn test_build_revisions_url_custom_base() {
        let url = build_revisions_url("test/model", "http://localhost:8080");
        assert_eq!(url, "http://localhost:8080/api/models/test/model/refs");
    }

    #[test]
    fn test_branch_names() {
        let json: Value = serde_json::from_str(
            r#"{
                "branches": [
                    {"name": "main"},
                    {"name": "gptq-4bit-64g-actorder_True"},
                    {"name": "develop"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(
            branch_names(&json),
            vec!["main", "gptq-4bit-64g-actorder_True", "develop"]
        );
    }

    #[test]
    fn test_branch_names_empty_array() {
        let json: Value = serde_json::from_str(r#"{"branches": []}"#).unwrap();
        assert_eq!(branch_names(&json), Vec::<String>::new());
    }

    #[test]
    fn test_branch_names_no_branches_field() {
        let json: Value = serde_json::from_str(r#"{"other_field": "value"}"#).unwrap();
        assert_eq!(branch_names(&json), Vec::<String>::new());
    }

    #[test]
    fn test_branch_names_missing_name() {
        let json: Value =
            serde_json::from_str(r#"{"branches": [{"invalid": "no name"}]}"#).unwrap();
        assert_eq!(branch_names(&json), Vec::<String>::new());
    }
}
