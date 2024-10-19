use std::error::Error;

pub fn download(
    repository: &String,
    revision: Option<&String>,
    to: Option<&std::path::PathBuf>,
    token: Option<&String>,
) -> Result<(), Box<dyn Error>> {
    println!(
        "Downloading repository: {}\n{}{}{}",
        repository,
        revision.map_or(String::from(""), |rev| format!("Revision: {}\n", rev)),
        to.map_or(String::from(""), |dir| format!("To: {}\n", dir.display())),
        token.map_or(String::from(""), |tok| format!("Token: {}\n", tok))
    );

    Ok(())
}
