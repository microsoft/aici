use std::process::Command;

fn run(cmd: &mut Command, msg: &str) {
    let status = cmd.status().expect(&format!("failed to execute: {msg}"));
    if !status.success() {
        panic!("process exited with status: {}; {}", status, msg);
    }
}

fn rerun_if_glob(pattern: &str) {
    for entry in glob::glob(pattern).expect(pattern).flatten() {
        let display = entry.display();
        println!("cargo:rerun-if-changed={display}");
    }
}

fn main() {
    rerun_if_glob("ts/*.ts");
    rerun_if_glob("ts/*.json");
    rerun_if_glob("gen-dts.mjs");

    if Command::new("tsc").arg("--version").status().is_err() {
        println!("cargo:warning=typescript not found, installing...");
        run(
            Command::new("npm")
                .arg("install")
                .arg("-g")
                .arg("typescript"),
            "npm install failed",
        );
    }

    run(Command::new("tsc").arg("-p").arg("ts"), "build failed");
    run(Command::new("node").arg("gen-dts.mjs"), "gen-dts failed");
}
