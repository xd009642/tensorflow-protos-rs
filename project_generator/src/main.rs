extern crate protoc_rust;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};
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

fn get_protos(tensorflow_dir: &Path) -> Vec<PathBuf> {
    let mut protos = vec![];

    let walker = WalkDir::new(tensorflow_dir).into_iter();
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
    protos
}

/// This assumption may break if protoc crate changes codegen details... Fun.
fn get_features_from_line(line: &str) -> HashSet<String> {
    let mut res = HashSet::new();
    let mut start = 0;
    let delimiter = "super::";
    loop {
        if let Some(index) = line[start..].find(delimiter) {
            let base = start + index + delimiter.len();
            if let Some(end) = line[base..].find("::") {
                res.insert(line[base..(base+end)].to_string());
                start = base+end+2;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    res
}

fn fill_in_features(src_dir: &Path) -> HashMap<String, HashSet<String>> {
    let mut feature_map = HashMap::new();
    let module_file = src_dir.join("protos.rs");
    let file = File::create(&module_file).expect("Failed to create src/protos.rs");
    let mut writer = BufWriter::new(file);

    for entry in WalkDir::new(src_dir.join("protos")).into_iter() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            let name = entry.path().file_stem().unwrap();
            let cfg = format!("#[cfg(feature = \"{}\")]\n", name.to_string_lossy());
            writer.write_all(cfg.as_bytes())
                .expect("Failed to write cfg attr");
            let source_line = format!("pub mod {};\n", name.to_string_lossy());
            writer
                .write_all(source_line.as_bytes())
                .expect("Failed to write to src/protos.rs");

            let mut dependencies = HashSet::new();
            let source = File::open(entry.path()).expect("Can't open source");
            for line in BufReader::new(source).lines() {
                let line = line.unwrap();
                for dep in get_features_from_line(&line).iter() {
                    dependencies.insert(dep.to_string());
                }
            }
            feature_map.insert(name.to_string_lossy().to_string(), dependencies);

        }
    }
    feature_map
}

fn generate_cargo_toml(features: HashMap<String, HashSet<String>>) {
    let file = File::create("Cargo.toml").expect("Failed to create Cargo.toml");
    let mut writer = BufWriter::new(file);
    
    let base_config = format!(r#"[package]
name = "tensorflow-protos-rs"
version = "{}"
authors = ["xd009642 <danielmckenna93@gmail.com>"]
edition = "2018"

[dependencies]
protobuf = "=2.17.0"

[features]
"#, env!("CARGO_PKG_VERSION"));

    writer.write_all(base_config.as_bytes()).unwrap();

    for (k, features) in features.iter() {
        writer.write_all(format!("{} = [", k).as_bytes()).unwrap();
        for feat in features {
            writer.write_all(format!("\"{}\",", feat).as_bytes()).unwrap();
        }
        writer.write_all(b"]\n").unwrap();
    }
}

fn main() {
    let tensorflow_dir = PathBuf::from("tensorflow");
    let src_dir = PathBuf::from("src");
    
    let protos = get_protos(&tensorflow_dir);
    
    assert!(!protos.is_empty(), "No proto files found!");

    protoc_rust::Codegen::new()
        .customize(Default::default())
        .out_dir(src_dir.join("protos"))
        .include(&tensorflow_dir)
        .inputs(&protos)
        .run()
        .expect("Code generation failed");

    let feature_map = fill_in_features(&src_dir);

    generate_cargo_toml(feature_map);
}

