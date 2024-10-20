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
    /// Do things with 🤗 models
    // e.g. cargo run --bin possum -- download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    /// Download weights
    Download {
        /// The model repository-id (e.g. TheBloke/Llama-2-7B-Chat-GPTQ)
        #[arg(long)]
        repository: String,

        /// Possibly a revision (e.g. gptq-4bit-64g-actorder_True)
        #[arg(short, long)]
        revision: Option<String>,

        /// A directory to download to (defaults to `./huggingface`)
        #[arg(short, long, default_value = DEFAULT_DOWNLOAD_DIR)]
        to: Option<std::path::PathBuf>,

        /// Huggingface token (sometimes required for 'gated' models)
        #[arg(long)]
        token: Option<String>,
    },
    /// Get metadata
    Metadata {
        /// The model repository-id (e.g. bert-base-uncased)
        #[arg(long)]
        repository: String,
    },
    /// Search based on keyword and filter
    // e.g. model search --keyword neuralmagic Llama-3.1 --filter gptq
    Search {
        /// Keywords for the search
        #[arg(long, num_args = 1..)]
        keyword: Vec<String>,

        /// An optional filter e.g. 'gptq' or 'text-classification'
        #[arg(long)]
        filter: Option<String>,
    },

    /// List the available revisions for a repository (tags or branches)
    //  e.g. cargo run --bin possum -- model revisions --repository TheBloke/Mixtral-8x7B-v0.1-GPTQ
    Revisions {
        /// The model repository-id (e.g. TheBloke/Mixtral-8x7B-v0.1-GPTQ)
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
        ModelCommands::Search { keyword, filter } => {
            commands::model::search(keyword, filter.as_deref()).await?;
        }
        ModelCommands::Revisions { repository } => {
            commands::model::revisions(repository).await?;
        }
    };

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();

    // cargo run --bin possum -- model search --keyword TheBloke Llama-2-7B --filter gptq
    // cargo run --bin possum -- model metadata --repository TheBloke/Llama-2-7B-Chat-GPTQ | jq '.transformersInfo'
    // cargo run --bin possum -- model revisions --repository TheBloke/Llama-2-7B-Chat-GPTQ
    // cargo run --bin possum -- model download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True

    match &args.command {
        Some(Commands::Model { command }) => model_command(command).await?,
        None => (),
    }

    Ok(())
}
