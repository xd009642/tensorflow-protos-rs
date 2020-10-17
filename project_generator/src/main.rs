extern crate protoc_rust;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
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
fn get_features_from_line(line: &str) -> BTreeSet<String> {
    let mut res = BTreeSet::new();
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

fn fill_in_features(gen_name: &str, src_dir: &Path, ignore_features: bool) -> BTreeMap<String, BTreeSet<String>> {
    let mut feature_map = BTreeMap::new();
    let module_file = src_dir.join(format!("{}.rs", gen_name));
    let file = File::create(&module_file).expect(&format!("Failed to create src/{}.rs", gen_name));
    let mut writer = BufWriter::new(file);

    for entry in WalkDir::new(src_dir.join(gen_name)).into_iter() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            let name = entry.path().file_stem().unwrap();
            let cfg = format!("#[cfg(feature = \"{}\")]\n", name.to_string_lossy());
            writer.write_all(cfg.as_bytes())
                .expect("Failed to write cfg attr");
            let source_line = format!("pub mod {};\n", name.to_string_lossy());
            writer
                .write_all(source_line.as_bytes())
                .expect(&format!("Failed to write to src/{}.rs", gen_name));

            let mut dependencies = BTreeSet::new();
            if !ignore_features {
                let source = File::open(entry.path()).expect("Can't open source");
                for line in BufReader::new(source).lines() {
                    let line = line.unwrap();
                    for dep in get_features_from_line(&line).iter() {
                        dependencies.insert(dep.to_string());
                    }
                }
            }
            feature_map.insert(name.to_string_lossy().to_string(), dependencies);

        }
    }
    feature_map
}

fn generate_cargo_toml(features: BTreeMap<String, BTreeSet<String>>) {
    let file = File::create("Cargo.toml").expect("Failed to create Cargo.toml");
    let mut writer = BufWriter::new(file);
    
    let base_config = format!(r#"[package]
name = "tensorflow-protos-rs"
version = "{}"
authors = ["xd009642 <danielmckenna93@gmail.com>"]
edition = "2018"

[dependencies]
prost = {{version ="0.6", optional = true}}
protobuf = {{version ="=2.17.0", optional = true}}

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

fn correct_filenames(folder: &Path) {
    for entry in WalkDir::new(folder).into_iter() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            let fname = entry.path().file_name().unwrap().to_str().unwrap();
            if fname.contains(".") {
                // inefficient but just want it to work for now
                let new_name = fname.replace(".", "_"); 
                let new_name = new_name.replace("_rs", ".rs");

                let fixed_name = entry.path().parent().unwrap().join(new_name);
                fs::rename(entry.path(), fixed_name).expect(&format!("Failed to fix name of {}", entry.path().display()));
            }
        }
    }
}

fn main() {
    let tensorflow_dir = PathBuf::from("tensorflow");
    let src_dir = PathBuf::from("src");
    
    let protos = get_protos(&tensorflow_dir);
    
    assert!(!protos.is_empty(), "No proto files found!");

    protoc_rust::Codegen::new()
        .customize(Default::default())
        .out_dir(src_dir.join("protobuf_gen"))
        .include(&tensorflow_dir)
        .inputs(&protos)
        .run()
        .expect("Protobuf code generation failed");

    let prost_outdir = src_dir.join("prost_gen");
    prost_build::Config::new()
        .out_dir(&prost_outdir)
        .compile_protos(&protos, &[tensorflow_dir])
        .expect("Prost code generation failed");

    correct_filenames(&prost_outdir);
    // Gonna need to correct file names

    let mut protobuf_feature_map = fill_in_features("protobuf_gen", &src_dir, false);
    let mut prost_feature_map = fill_in_features("prost_gen", &src_dir, true);
    let mut feature_map = BTreeMap::new();

    for (k, v) in protobuf_feature_map.iter_mut() {
        if !prost_feature_map.contains_key(k) {
            v.insert(String::from("protobuf"));
        }
        feature_map.insert(k.clone(), v.clone());
    }

    for (k, v) in prost_feature_map.iter_mut() {
        if !protobuf_feature_map.contains_key(k) {
            v.insert(String::from("prost"));
            feature_map.insert(k.clone(), v.clone());
        }
    }

    generate_cargo_toml(feature_map);
}

