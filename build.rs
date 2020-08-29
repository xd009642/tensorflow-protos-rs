extern crate protoc_rust;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Component, PathBuf};
use walkdir::{DirEntry, WalkDir};

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with("."))
        .unwrap_or(false)
}

fn is_bad_path(c: Component<'_>) -> bool {
    c == Component::Normal("contrib".as_ref())
        || c == Component::Normal("example".as_ref())
        || c == Component::Normal("python".as_ref())
        || c == Component::Normal("lite".as_ref())
}

/// Life is easier if you pretend tensorflow contrib modules aren't a thing in civilised society.
fn ignored_proto_source(entry: &DirEntry) -> bool {
    entry.path().components().into_iter().any(is_bad_path)
}

fn main() {
    let tensorflow_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tensorflow");
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");

    let mut protos = vec![];

    let walker = WalkDir::new(&tensorflow_dir).into_iter();
    for entry in walker.filter_entry(|e| !(is_hidden(e) || ignored_proto_source(e))) {
        let entry = entry.expect("Invalid DirEntry found");
        let path = entry.path();
        match path.to_str() {
            Some(s) => {
                // There seems to be a single unit test proto hidden in the source tree.
                if s.contains("example") || s.contains("test") {
                    continue;
                }
            }
            None => continue,
        }
        if path.extension().unwrap_or_default() == "proto" && path.is_file() {
            protos.push(path.to_path_buf());
        }
    }

    assert!(!protos.is_empty(), "No proto files found!");

    protoc_rust::Codegen::new()
        .customize(Default::default())
        .out_dir(src_dir.join("protos"))
        .include(&tensorflow_dir)
        .inputs(&protos)
        .run()
        .expect("Code generation failed");

    let module_file = src_dir.join("protos.rs");
    let file = File::create(&module_file).expect("Failed to create src/protos.rs");
    let mut writer = BufWriter::new(file);

    for entry in WalkDir::new(src_dir.join("protos")).into_iter() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            let name = entry.path().file_stem().unwrap();
            let source_line = format!("pub mod {};\n", name.to_string_lossy());
            writer
                .write_all(source_line.as_bytes())
                .expect("Failed to write to src/protos.rs");
        }
    }
}
