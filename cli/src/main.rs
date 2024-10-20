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

async fn model_command(command: &ModelCommands) -> Result<(), Box<dyn Error + Send + Sync>> {
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
            commands::model::download(repository, revision.as_ref(), &local_dir, token.as_ref())
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
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    // cargo run --bin possum -- model search --keyword TheBloke Llama-2-7B --filter gptq
    // cargo run --bin possum -- model metadata --repository TheBloke/Llama-2-7B-Chat-GPTQ | jq '.transformersInfo'
    // cargo run --bin possum -- model revisions --repository TheBloke/Llama-2-7B-Chat-GPTQ
    // cargo run --bin possum -- model download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True

    tracing::info!("Hello possums! ✨");
    match &args.command {
        Some(Commands::Model { command }) => model_command(command).await?,
        None => (),
    }

    Ok(())
}
