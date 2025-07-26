mod commands;

use clap::{Parser, Subcommand};

use std::error::Error;

const DEFAULT_DOWNLOAD_DIR: &str = "./huggingface";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Base URL for HuggingFace API (for testing)
    #[arg(
        long,
        env = "HUGGINGFACE_API_BASE_URL",
        default_value = "https://huggingface.co"
    )]
    api_base_url: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Do things with 🤗 models
    // e.g. cargo run --bin possum -- download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    /// Download repository files
    Download {
        /// The model repository-id (e.g. TheBloke/Llama-2-7B-Chat-GPTQ)
        #[arg(long)]
        repository: String,

        /// Optional revision (e.g. gptq-4bit-64g-actorder_True)
        #[arg(short, long)]
        revision: Option<String>,

        /// A directory to download to (default: `./huggingface`)
        #[arg(short, long, default_value = DEFAULT_DOWNLOAD_DIR)]
        to: Option<std::path::PathBuf>,

        /// Hugging Face token (might be needed for 'gated' models)
        #[arg(long)]
        token: Option<String>,
    },
    /// Get repository metadata
    Metadata {
        /// The model repository-id (e.g. TheBloke/Llama-2-7B-Chat-GPTQ)
        #[arg(long)]
        repository: String,
    },
    /// Search for repositories based on keywords and a filter
    Search {
        /// Keywords for the search
        #[arg(long, num_args = 1..)]
        keyword: Vec<String>,

        /// An optional filter (e.g. 'gptq' or 'text-classification')
        #[arg(long)]
        filter: Option<String>,
    },

    /// List available revisions of a repository
    Revisions {
        /// The model repository-id (e.g. TheBloke/Llama-2-7B-Chat-GPTQ)
        #[arg(long)]
        repository: String,
    },
}

async fn model_command(
    command: &ModelCommands,
    api_base_url: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match command {
        ModelCommands::Download {
            repository,
            revision,
            to,
            token,
        } => {
            let mut local_dir = to.as_ref().unwrap().clone();
            local_dir.push(repository);
            if let Some(rev) = revision {
                // Convert to a string to append revision
                let lds = local_dir.to_string_lossy();
                local_dir = std::path::PathBuf::from(format!("{}:{}", lds, rev));
            }
            commands::model::download(
                repository,
                revision.as_ref(),
                &local_dir,
                token.as_ref(),
                api_base_url,
            )
            .await?
        }
        ModelCommands::Metadata { repository } => {
            commands::model::metadata(repository, api_base_url).await?;
        }
        ModelCommands::Search { keyword, filter } => {
            commands::model::search(keyword, filter.as_deref(), api_base_url).await?;
        }
        ModelCommands::Revisions { repository } => {
            commands::model::revisions(repository, api_base_url).await?;
        }
    };

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    // cargo run --bin possum -- model search --keyword TheBloke Llama-2-7B --filter gptq
    // cargo run --bin possum -- model metadata --repository TheBloke/Llama-2-7B-Chat-GPTQ | jq '.transformersInfo'
    // cargo run --bin possum -- model revisions --repository TheBloke/Llama-2-7B-Chat-GPTQ
    // cargo run --bin possum -- model download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True

    tracing::info!("Hello possums! ✨");
    match &args.command {
        Some(Commands::Model { command }) => model_command(command, &args.api_base_url).await?,
        None => (),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_model_search() {
        let args = Args::parse_from([
            "possum",
            "model",
            "search",
            "--keyword",
            "TheBloke",
            "--keyword",
            "Llama-2-7B",
            "--filter",
            "gptq",
        ]);

        match args.command {
            Some(Commands::Model {
                command: ModelCommands::Search { keyword, filter },
            }) => {
                assert_eq!(keyword, vec!["TheBloke", "Llama-2-7B"]);
                assert_eq!(filter, Some("gptq".to_string()));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_model_metadata() {
        let args = Args::parse_from([
            "possum",
            "model",
            "metadata",
            "--repository",
            "TheBloke/Llama-2-7B-Chat-GPTQ",
        ]);

        match args.command {
            Some(Commands::Model {
                command: ModelCommands::Metadata { repository },
            }) => {
                assert_eq!(repository, "TheBloke/Llama-2-7B-Chat-GPTQ");
            }
            _ => panic!("Expected Metadata command"),
        }
    }

    #[test]
    fn test_cli_model_revisions() {
        let args = Args::parse_from([
            "possum",
            "model",
            "revisions",
            "--repository",
            "TheBloke/Llama-2-7B-Chat-GPTQ",
        ]);

        match args.command {
            Some(Commands::Model {
                command: ModelCommands::Revisions { repository },
            }) => {
                assert_eq!(repository, "TheBloke/Llama-2-7B-Chat-GPTQ");
            }
            _ => panic!("Expected Revisions command"),
        }
    }

    #[test]
    fn test_cli_model_download() {
        let args = Args::parse_from([
            "possum",
            "model",
            "download",
            "--repository",
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            "--revision",
            "gptq-4bit-64g-actorder_True",
        ]);

        match args.command {
            Some(Commands::Model {
                command:
                    ModelCommands::Download {
                        repository,
                        revision,
                        to,
                        token,
                    },
            }) => {
                assert_eq!(repository, "TheBloke/Llama-2-7B-Chat-GPTQ");
                assert_eq!(revision, Some("gptq-4bit-64g-actorder_True".to_string()));
                assert_eq!(to, Some(std::path::PathBuf::from(DEFAULT_DOWNLOAD_DIR)));
                assert_eq!(token, None);
            }
            _ => panic!("Expected Download command"),
        }
    }

    #[test]
    fn test_cli_model_download_with_custom_dir() {
        let args = Args::parse_from([
            "possum",
            "model",
            "download",
            "--repository",
            "TheBloke/Llama-2-7B-Chat-GPTQ",
            "--to",
            "/custom/path",
        ]);

        match args.command {
            Some(Commands::Model {
                command:
                    ModelCommands::Download {
                        repository,
                        revision,
                        to,
                        token,
                    },
            }) => {
                assert_eq!(repository, "TheBloke/Llama-2-7B-Chat-GPTQ");
                assert_eq!(revision, None);
                assert_eq!(to, Some(std::path::PathBuf::from("/custom/path")));
                assert_eq!(token, None);
            }
            _ => panic!("Expected Download command"),
        }
    }
}
