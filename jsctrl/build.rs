use std::process::Command;

fn run(cmd: &mut Command, msg: &str) {
    let status = cmd.status().expect(&format!("failed to execute: {msg}"));
    if !status.success() {
        panic!("process exited with status: {}; {}", status, msg);
    }
}

fn main() {
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
