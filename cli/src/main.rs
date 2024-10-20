mod commands;

use clap::{Parser, Subcommand};

use std::error::Error;

const DEFAULT_DOWNLOAD_DIR: &str = "./huggingface";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a Hugging Face model
    // e.g. cargo run --bin possum -- download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    Download {
        /// The model repository-id
        #[arg(long)]
        repository: String,

        /// An optional revision
        #[arg(short, long)]
        revision: Option<String>,

        /// An optional local directory (default)
        #[arg(short, long, default_value = DEFAULT_DOWNLOAD_DIR)]
        to: Option<std::path::PathBuf>,

        #[arg(long)]
        token: Option<String>,
    },
    /// Get metadata for a Hugging Face model
    Metadata {
        /// The model repository-id (e.g. bert-base-uncased)
        #[arg(long)]
        repository: String,
    },
}

async fn model_command(command: &ModelCommands) -> Result<(), Box<dyn Error + Send + Sync>> {
    match command {
        ModelCommands::Download {
            repository,
            revision,
            to,
            token,
        } => {
            commands::model::download(
                repository,
                revision.as_ref(),
                to.as_ref().unwrap(),
                token.as_ref(),
            )
            .await?
        }
        ModelCommands::Metadata { repository } => {
            commands::model::metadata(repository).await?;
        }
    };

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();

    // cargo run -p cli -- model download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True --token abc

    match &args.command {
        Some(Commands::Model { command }) => model_command(command).await?,
        None => (),
    }

    Ok(())
}
