use reqwest::Client;
use serde_json::Value;
use std::error::Error;

fn extract_branch_names(parsed_json: &Value) -> Result<(), Box<dyn Error + Send + Sync>> {
    if let Some(branches) = parsed_json.get("branches").and_then(|b| b.as_array()) {
        for branch in branches {
            if let Some(name) = branch.get("name").and_then(|n| n.as_str()) {
                println!("{name}");
            }
        }
    } else {
        println!("No branches found.");
    }

    Ok(())
}

/// Function to discover revisions (branches or tags) for a Hugging
/// Face repository
pub async fn revisions(
    repository: &str,
    api_base_url: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let url = build_revisions_url(repository, api_base_url);

    let response = client.get(&url).send().await?;

    if response.status().is_success() {
        let refs: Value = response.json().await?;
        extract_branch_names(&refs)?;

        Ok(())
    } else {
        println!("Failed to fetch refs: {}", response.status());
        Err(format!("Failed to fetch refs for '{repository}'").into())
    }
}

pub fn build_revisions_url(repository: &str, api_base_url: &str) -> String {
    format!("{api_base_url}/api/models/{repository}/refs")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_branches_from_json(
        json_str: &str,
    ) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
        let parsed: Value = serde_json::from_str(json_str)?;
        let mut branches = Vec::new();

        if let Some(branch_array) = parsed.get("branches").and_then(|b| b.as_array()) {
            for branch in branch_array {
                if let Some(name) = branch.get("name").and_then(|n| n.as_str()) {
                    branches.push(name.to_string());
                }
            }
        }

        Ok(branches)
    }

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
    fn test_parse_branches_from_json() {
        let json = r#"{
            "branches": [
                {"name": "main"},
                {"name": "gptq-4bit-64g-actorder_True"},
                {"name": "develop"}
            ]
        }"#;

        let branches = parse_branches_from_json(json).unwrap();
        assert_eq!(
            branches,
            vec!["main", "gptq-4bit-64g-actorder_True", "develop"]
        );
    }

    #[test]
    fn test_parse_branches_empty_array() {
        let json = r#"{"branches": []}"#;
        let branches = parse_branches_from_json(json).unwrap();
        assert_eq!(branches, Vec::<String>::new());
    }

    #[test]
    fn test_parse_branches_no_branches_field() {
        let json = r#"{"other_field": "value"}"#;
        let branches = parse_branches_from_json(json).unwrap();
        assert_eq!(branches, Vec::<String>::new());
    }

    #[test]
    fn test_parse_branches_invalid_json() {
        let json = r#"{"branches": [{"invalid": "no name field"}]}"#;
        let branches = parse_branches_from_json(json).unwrap();
        assert_eq!(branches, Vec::<String>::new());
    }
}
