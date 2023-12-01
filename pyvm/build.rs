fn main() {
    for entry in glob::glob("Lib/**/*.py").expect("Lib/ exists?").flatten() {
        let display = entry.display();
        println!("cargo:rerun-if-changed={display}");
    }
}
