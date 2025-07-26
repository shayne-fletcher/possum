use assert_cmd::Command;
use serde_json::json;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_model_search_integration() {
    // Start a mock server
    let mock_server = MockServer::start().await;

    // Mock the HuggingFace API response
    let mock_response = json!([
        {"modelId": "TheBloke/Llama-2-7B-Chat-GPTQ"},
        {"modelId": "TheBloke/Llama-2-13B-Chat-GPTQ"}
    ]);

    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("search", "TheBloke Llama-2-7B"))
        .and(query_param("filter", "gptq"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    // Now we can use the mock server URL
    let mut cmd = Command::cargo_bin("possum").unwrap();
    let output = cmd
        .args(&[
            "--api-base-url",
            &mock_server.uri(),
            "model",
            "search",
            "--keyword",
            "TheBloke",
            "--keyword",
            "Llama-2-7B",
            "--filter",
            "gptq",
        ])
        .output()
        .unwrap();

    // Check that the command succeeded
    assert!(output.status.success());

    // Check that the output contains the expected model IDs
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("TheBloke/Llama-2-7B-Chat-GPTQ"));
    assert!(stdout.contains("TheBloke/Llama-2-13B-Chat-GPTQ"));
}

#[tokio::test]
async fn test_model_metadata_integration() {
    let mock_server = MockServer::start().await;

    let mock_response = json!({
        "modelId": "TheBloke/Llama-2-7B-Chat-GPTQ",
        "sha": "abc123",
        "transformersInfo": {
            "auto_model": "AutoModelForCausalLM"
        }
    });

    Mock::given(method("GET"))
        .and(path("/api/models/TheBloke/Llama-2-7B-Chat-GPTQ"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    let mut cmd = Command::cargo_bin("possum").unwrap();
    let output = cmd
        .args(&[
            "--api-base-url",
            &mock_server.uri(),
            "model",
            "metadata",
            "--repository",
            "TheBloke/Llama-2-7B-Chat-GPTQ",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("TheBloke/Llama-2-7B-Chat-GPTQ"));
    assert!(stdout.contains("AutoModelForCausalLM"));
}

#[tokio::test]
async fn test_model_revisions_integration() {
    let mock_server = MockServer::start().await;

    let mock_response = json!({
        "branches": [
            {"name": "main"},
            {"name": "gptq-4bit-64g-actorder_True"}
        ]
    });

    Mock::given(method("GET"))
        .and(path("/api/models/TheBloke/Llama-2-7B-Chat-GPTQ/refs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    let mut cmd = Command::cargo_bin("possum").unwrap();
    let output = cmd
        .args(&[
            "--api-base-url",
            &mock_server.uri(),
            "model",
            "revisions",
            "--repository",
            "TheBloke/Llama-2-7B-Chat-GPTQ",
        ])
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("main"));
    assert!(stdout.contains("gptq-4bit-64g-actorder_True"));
}

#[tokio::test]
async fn test_model_download_integration() {
    let mock_server = MockServer::start().await;

    // Mock the file listing response
    let mock_response = json!({
        "siblings": [
            {"rfilename": "config.json"},
            {"rfilename": "model.safetensors"}
        ]
    });

    Mock::given(method("GET"))
        .and(path("/api/models/test/model"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&mock_response))
        .mount(&mock_server)
        .await;

    // Mock file downloads
    Mock::given(method("GET"))
        .and(path("/test/model/resolve/main/config.json"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/test/model/resolve/main/model.safetensors"))
        .respond_with(ResponseTemplate::new(200).set_body_string("fake model data"))
        .mount(&mock_server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();

    let mut cmd = Command::cargo_bin("possum").unwrap();
    let output = cmd
        .args(&[
            "--api-base-url",
            &mock_server.uri(),
            "model",
            "download",
            "--repository",
            "test/model",
            "--to",
            temp_dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(output.status.success());

    // Check that files were created
    let model_dir = temp_dir.path().join("test").join("model");
    assert!(model_dir.join("config.json").exists());
    assert!(model_dir.join("model.safetensors").exists());
}

#[test]
fn test_cli_help_output() {
    let mut cmd = Command::cargo_bin("possum").unwrap();
    let output = cmd.arg("--help").output().unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Do things with 🤗 models"));
}

#[test]
fn test_cli_version_output() {
    let mut cmd = Command::cargo_bin("possum").unwrap();
    let output = cmd.arg("--version").output().unwrap();

    assert!(output.status.success());
}
