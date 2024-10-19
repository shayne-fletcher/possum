mod commands;

use clap::{Parser, Subcommand};

use std::error::Error;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

const DEFAULT_TO: &str = "./huggingface";

#[derive(Subcommand)]
enum Commands {
    /// Download a Hugging Face model
    // e.g. cargo run -p cli -- download --repository TheBloke/Llama-2-7B-Chat-GPTQ --revision gptq-4bit-64g-actorder_True
    Download {
        /// The model repository-id
        #[arg(long)]
        repository: String,

        /// An optional revision
        #[arg(short, long)]
        revision: Option<String>,

        /// An optional local directory (default)
        #[arg(short, long, default_value = DEFAULT_TO)]
        to: Option<std::path::PathBuf>,

        #[arg(long)]
        token: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match &args.command {
        Some(Commands::Download {
            repository,
            revision,
            to,
            token,
        }) => {
            commands::download(repository, revision.as_ref(), to.as_ref(), token.as_ref())?;
        }
        None => (),
    }

    Ok(())
}
