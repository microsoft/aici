fn main() {
    for entry in glob::glob("Lib/**/*.py").expect("Lib/ exists?").flatten() {
        let display = entry.display();
        println!("cargo:rerun-if-changed={display}");
    }
    for entry in glob::glob("aici-pylib/**/*.py").expect("aici-pylib/ exists?").flatten() {
        let display = entry.display();
        println!("cargo:rerun-if-changed={display}");
    }
}
