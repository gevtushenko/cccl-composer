use clap::{Arg, ArgAction, ArgMatches, Command};
use clap_complete::{generate, shells::Zsh};
use config::{Config, ConfigError, File};
use dirs::config_dir;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use prettytable::{color, Attr, Cell, Row, Table};
use rayon;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command as ProcCommand;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use colored::*;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CompilerConfig {
    label: String,
    version: String,
    path: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CTKConfig {
    label: String,
    path: String,
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    src: HashMap<String, String>,
    compilers: HashMap<String, String>,
    ctks: HashMap<String, String>,
}

impl AppConfig {
    fn compiler_labels(&self) -> Vec<&str> {
        return self.compilers.keys().map(String::as_str).collect();
    }

    fn ctk_labels(&self) -> Vec<&str> {
        return self.ctks.keys().map(String::as_str).collect();
    }
}

impl AppConfig {
    pub fn new() -> Result<Self, ConfigError> {
        let config_path = config_dir().unwrap();
        let cccl_config_path = config_path.join("cccl-composer").join("config.json");
        let s = Config::builder()
            .add_source(File::with_name(cccl_config_path.to_str().unwrap()))
            .build()?;

        s.try_deserialize()
    }
}

fn build_cli(config: &AppConfig) -> clap::App {
    let compilers: Vec<&str> = config.compiler_labels();
    let ctks: Vec<&str> = config.ctk_labels();

    return Command::new("cccl-composer")
        .about("cccl infrastructure utility")
        .version("0.0.1")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .author("CUDA C++ Core Libraries Team")
        .subcommand(
            Command::new("run")
                .short_flag('r')
                .long_flag("run")
                .about("Run CUB tests.")
                .arg(
                    Arg::new("compilers")
                        .short('c')
                        .long("compilers")
                        .action(ArgAction::Set)
                        .multiple_values(true)
                        .help("specify compilers."),
                )
                .arg(
                    Arg::new("ctk")
                        .long("cuda")
                        .action(ArgAction::Set)
                        .multiple_values(true)
                        .help("specify CTK versions."),
                )
                .arg(
                    Arg::new("targets")
                        .help("targets")
                        .action(ArgAction::Set)
                        .multiple_values(true),
                ),
        )
        .subcommand(
            Command::new("build")
                .short_flag('S')
                .long_flag("build")
                .about("Build CUB tests.")
                .arg(
                    Arg::new("compilers")
                        .short('c')
                        .long("compilers")
                        .action(ArgAction::Set)
                        .multiple_values(true)
                        .possible_values(compilers)
                        .help("specify compilers."),
                )
                .arg(
                    Arg::new("dialects")
                        .short('d')
                        .long("dialects")
                        .action(ArgAction::Set)
                        .multiple_values(true)
                        .possible_values(["11", "14", "17"])
                        .help("specify C++ dialects."),
                )
                .arg(
                    Arg::new("types")
                        .short('t')
                        .long("types")
                        .action(ArgAction::Set)
                        .multiple_values(true)
                        .possible_values(["debug", "release"])
                        .help("specify build types."),
                )
                .arg(
                    Arg::new("ctks")
                        .long("ctks")
                        .action(ArgAction::Set)
                        .multiple_values(true)
                        .possible_values(ctks)
                        .help("specify CTK versions."),
                )
                .arg(
                    Arg::new("targets")
                        .long("targets")
                        .help("targets")
                        .action(ArgAction::Set)
                        .multiple_values(true),
                ),
        )
        .subcommand(Command::new("generate-zsh-completions").about("Generate Zsh completions."));
}

fn get_compilers<'a>(config: &'a AppConfig, matches: &'a ArgMatches) -> Vec<&'a str> {
    if matches.contains_id("compilers") {
        return matches
            .get_many::<String>("compilers")
            .expect("contains_id")
            .map(|s| s.as_str())
            .collect();
    } else {
        return config.compiler_labels();
    }
}

fn get_build_types(matches: &ArgMatches) -> Vec<&str> {
    if matches.contains_id("types") {
        return matches
            .get_many::<String>("types")
            .expect("contains_id")
            .map(|s| s.as_str())
            .collect();
    } else {
        return vec!["debug", "release"];
    }
}

fn get_ctks<'a>(config: &'a AppConfig, matches: &'a ArgMatches) -> Vec<&'a str> {
    if matches.contains_id("ctks") {
        return matches
            .get_many::<String>("ctks")
            .expect("contains_id")
            .map(|s| s.as_str())
            .collect();
    } else {
        return config.ctk_labels();
    }
}

fn get_dialects(matches: &ArgMatches) -> Vec<&str> {
    if matches.contains_id("dialects") {
        return matches
            .get_many::<String>("dialects")
            .expect("contains_id")
            .map(|s| s.as_str())
            .collect();
    } else {
        return vec!["11", "14", "17"];
    }
}

fn get_targets(cpp: &Vec<&str>, matches: &ArgMatches) -> HashMap<String, String> {
    let mut result: HashMap<String, String> = HashMap::new();

    if matches.contains_id("targets") {
        let targets: Vec<_> = matches
            .get_many::<String>("targets")
            .expect("is present")
            .map(|s| s.as_str())
            .collect();

        for target in targets.iter() {
            for dialect in cpp {
                result.insert(
                    dialect.to_string(),
                    format!("cub.cpp{}.{}", dialect, target),
                );
            }
        }
    } else {
        for dialect in cpp {
            result.insert(dialect.to_string(), String::from(""));
        }
    }

    return result;
}

#[derive(Debug)]
struct BuildResult<'a> {
    data: HashMap<&'a str, HashMap<&'a str, HashMap<&'a str, HashMap<&'a str, bool>>>>,
}

impl<'a> BuildResult<'a> {
    fn new(
        types: &Vec<&'a str>,
        ctks: &Vec<&'a str>,
        cpp: &Vec<&'a str>,
        compilers: &Vec<&'a str>,
    ) -> Self {
        let mut compilers_state: Vec<(&'a str, bool)> = Vec::new();
        for compiler in compilers {
            compilers_state.push((compiler, false));
        }
        let compilers_state: HashMap<&'a str, bool> = compilers_state.into_iter().collect();

        let mut cpp_state: HashMap<&'a str, HashMap<&'a str, bool>> = HashMap::new();
        for dialect in cpp {
            cpp_state.insert(dialect, compilers_state.clone());
        }

        let mut ctk_state: HashMap<&'a str, HashMap<&'a str, HashMap<&'a str, bool>>> =
            HashMap::new();
        for ctk in ctks {
            ctk_state.insert(ctk, cpp_state.clone());
        }

        let mut type_state: HashMap<
            &'a str,
            HashMap<&'a str, HashMap<&'a str, HashMap<&'a str, bool>>>,
        > = HashMap::new();
        for build_type in types {
            type_state.insert(build_type, ctk_state.clone());
        }

        return Self { data: type_state };
    }

    fn success(&mut self, build_type: &'a str, ctk: &'a str, cpp: &'a str, compiler: &'a str) {
        *self
            .data
            .get_mut(build_type)
            .unwrap()
            .get_mut(ctk)
            .unwrap()
            .get_mut(cpp)
            .unwrap()
            .get_mut(compiler)
            .unwrap() = true;
    }

    fn status(&self, build_type: &'a str, ctk: &'a str, cpp: &'a str, compiler: &'a str) -> ColoredString {
        if *self
            .data
            .get(build_type)
            .unwrap()
            .get(ctk)
            .unwrap()
            .get(cpp)
            .unwrap()
            .get(compiler)
            .unwrap()
        {
            return "✓".green();
        } else {
            return "✗".red();
        }
    }
}

fn build(config: &AppConfig, matches: &ArgMatches) {
    let types = get_build_types(&matches);
    let compilers = get_compilers(&config, &matches);
    let ctks = get_ctks(&config, &matches);
    let cpp = get_dialects(matches);
    let targets = get_targets(&cpp, matches);

    let num_builds = ctks.len() * compilers.len() * cpp.len() * types.len();
    let results = Arc::new(Mutex::new(BuildResult::new(
        &types, &ctks, &cpp, &compilers,
    )));

    let num_cpus = std::thread::available_parallelism().unwrap().get();
    let num_concurrent_builds = std::cmp::min(num_cpus, num_builds);
    let num_threads_per_build = num_cpus / num_concurrent_builds;

    println!("Build with {num_threads_per_build} threads per build and {num_concurrent_builds} concurrent builds");

    rayon::scope(|s| {
        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        for ctk in &ctks {
            for compiler_label in &compilers {
                for dialect in &cpp {
                    for build_type in &types {
                        let pb = m.add(ProgressBar::new(cpp.len() as u64));
                        pb.set_style(sty.clone());
                        pb.set_position(0);

                        let compiler = compiler_label.replace("/", ".");
                        pb.set_message(format!(
                            "{}/{}/{}/cpp.{}",
                            build_type, ctk, compiler, dialect
                        ));

                        s.spawn(|_| {
                            let cxx_path = config
                                .compilers
                                .get(&compiler_label.to_string())
                                .unwrap()
                                .clone();

                            let pb = pb;
                            let compiler_label = compiler_label.clone();
                            let compiler = compiler_label.replace("/", ".");
                            let ctk = ctk.clone();
                            let result = Arc::clone(&results);
                            let ctk_path = config.ctks.get(&ctk.to_string()).unwrap().clone();
                            let build_type = build_type.clone();
                            let dialect = dialect.clone();

                            let cub_path = config.src.get("cub").unwrap();
                            let thrust_path = config.src.get("thrust").unwrap();
                            let nvcc_path = Path::new(&ctk_path).join("bin").join("nvcc");
                            let nvcc_path_str = nvcc_path.to_str().unwrap().clone();
                            let current_dir = env::current_dir().unwrap();
                            let mut build_dir = current_dir.clone();

                            build_dir.push("build");
                            build_dir.push(ctk.clone());
                            build_dir.push(&build_type);
                            build_dir.push(&compiler);
                            build_dir.push(&dialect);

                            fs::create_dir_all(&build_dir).ok();
                            let build_dir = build_dir.into_os_string().into_string().unwrap();

                            let mut arguments: Vec<String> = Vec::new();

                            arguments.push("-GNinja".to_string());
                            arguments.push(format!("-B{}", build_dir));
                            arguments
                                .push(format!("-DCMAKE_BUILD_TYPE={}", build_type).to_string());
                            arguments.push("-DCUB_DISABLE_ARCH_BY_DEFAULT=ON".to_string());
                            arguments.push("-DCUB_ENABLE_COMPUTE_80=ON".to_string());
                            arguments.push("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT=ON".to_string());
                            arguments.push("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON".to_string());

                            if compiler.contains("nvhpc") {
                                // TODO Push ctk version
                                // TODO -DCMAKE_CUDA_FLAGS="-gpu=cuda11.6 -gpu=cc86"
                                arguments.push("-DCMAKE_CUDA_COMPILER_FORCED=ON".to_string());
                                arguments.push(
                                    format!("-DCMAKE_CUDA_COMPILER={}", &cxx_path).to_string(),
                                );
                                arguments.push("-DCMAKE_CUDA_COMPILER_ID=NVCXX".to_string());
                            } else {
                                arguments.push(
                                    format!("-DCMAKE_CUDA_COMPILER={}", nvcc_path_str).to_string(),
                                );
                                arguments.push(format!("-DCMAKE_CXX_COMPILER={}", &cxx_path));
                            }

                            for d in ["11", "14", "17"] {
                                if d == dialect {
                                    arguments.push(
                                        format!("-DCUB_ENABLE_DIALECT_CPP{}=ON", d.to_string())
                                            .to_string(),
                                    );
                                } else {
                                    arguments.push(
                                        format!("-DCUB_ENABLE_DIALECT_CPP{}=OFF", d.to_string())
                                            .to_string(),
                                    );
                                }
                            }
                            arguments.push(
                                format!("-DThrust_DIR={}/thrust/cmake", thrust_path).to_string(),
                            );
                            arguments.push("-DCUB_ENABLE_TESTS_WITH_RDC=OFF".to_string());
                            arguments.push(cub_path.clone());

                            let cmake_output = ProcCommand::new("cmake")
                                .args(arguments)
                                .output()
                                .expect("failed to execute cmake process");

                            let mut results: Vec<String> = Vec::new();

                            results.push(compiler.to_string());

                            if !cmake_output.status.success() {
                                println!(
                                    "stderr 1: {}",
                                    String::from_utf8_lossy(&cmake_output.stderr)
                                );
                                results.push("-".to_string());
                                return;
                            }

                            let re = Regex::new(r"^\[(?P<current>\d+)/(?P<total>\d+)\]").unwrap();

                            let mut arguments: Vec<String> = Vec::new();
                            arguments.push(format!("-C{}", &build_dir).to_string());
                            arguments.push(format!("-j{}", num_threads_per_build).to_string());

                            let tgt = targets.get(&dialect.to_string()).unwrap();

                            if !tgt.is_empty() {
                                arguments.push(tgt.to_string());
                            }

                            let mut ninja_child = ProcCommand::new("ninja")
                                .args(arguments)
                                .stdout(Stdio::piped())
                                .spawn()
                                .expect("failed to execute ninja process");

                            loop {
                                {
                                    let mut f =
                                        BufReader::new(ninja_child.stdout.as_mut().unwrap());
                                    let mut buf = String::new();
                                    match f.read_line(&mut buf) {
                                        Ok(_) => {
                                            if buf.is_empty() {
                                                // println!("empty line, exit");
                                            } else {
                                                match re.captures(&buf) {
                                                    Some(caps) => {
                                                        let current: u64 =
                                                            caps["current"].parse().unwrap();
                                                        let total: u64 =
                                                            caps["total"].parse().unwrap();
                                                        pb.set_length(total);
                                                        pb.set_position(current);
                                                    }
                                                    None => {}
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            println!("an error!: {:?}", e);
                                            break;
                                        }
                                    }
                                }

                                match ninja_child.try_wait() {
                                    Ok(Some(status)) => {
                                        // println!("exited with: {status}");

                                        if !status.success() {
                                            results.push("✗".to_string());
                                        }
                                        break;
                                    }
                                    Ok(None) => {
                                        // println!("status not ready yet");
                                    }
                                    Err(e) => println!("error attempting to wait: {e}"),
                                }
                            }

                            pb.finish();

                            let mut r = result.lock().unwrap();
                            r.success(&build_type, &ctk, &dialect, &compiler_label);
                        });
                    }
                }
            }
        }

        m.clear().unwrap();
    });

    let result = results.lock().unwrap();

    let mut summary_table: Table = Table::new();

    let mut build_row: Vec<Table> = Vec::new();
    for build_type in &types {
        let mut ctk_row: Vec<Table> = Vec::new();
        for ctk in &ctks {
            let mut cpp_row: Vec<Table> = Vec::new();
            for dialect in &cpp {
                let mut compiler_table: Table = Table::new();
                for compiler in &compilers {
                    compiler_table.add_row(Row::from([
                        compiler.clear(),
                        result.status(build_type, ctk, dialect, compiler),
                    ]));
                }
                cpp_row.push(compiler_table);
            }
            let mut cpp_table: Table = Table::new();
            cpp_table.add_row(Row::from(cpp.iter().map(|str| {str.yellow().bold()}).collect::<Vec<ColoredString>>()));
            cpp_table.add_row(Row::from(cpp_row));
            ctk_row.push(cpp_table);
        }
        let mut ctk_table: Table = Table::new();
        ctk_table.add_row(Row::from(ctks.iter().map(|str| {str.yellow().bold()}).collect::<Vec<ColoredString>>()));
        ctk_table.add_row(Row::from(ctk_row));
        build_row.push(ctk_table);
    }
    summary_table.add_row(Row::from(types.iter().map(|str| {str.yellow().bold()}).collect::<Vec<ColoredString>>()));
    summary_table.add_row(Row::from(build_row));
    summary_table.printstd();
}

fn main() -> std::io::Result<()> {
    let maybe_config = AppConfig::new();

    match maybe_config {
        Ok(config) => {
            let matches = build_cli(&config).get_matches();

            match matches.subcommand() {
                Some(("build", build_matches)) => {
                    build(&config, &build_matches);
                }
                Some(("generate-zsh-completions", _)) => {
                    generate(
                        Zsh,
                        &mut build_cli(&config),
                        "cccl-composer",
                        &mut io::stdout(),
                    );
                }
                Some(("run", _)) => {
                    println!("run is unsupported");
                }
                _ => unreachable!(), // If all subcommands are defined above, anything else is unreachable
            }
        }
        _ => {
            println!("configuration loading error");
        }
    }

    Ok(())
}
