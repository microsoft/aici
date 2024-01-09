use rquickjs::{Context, Function, Runtime};

fn print(msg: String) {
    println!("{msg}");
}

fn main() {
    let rt = Runtime::new().unwrap();
    let ctx = Context::full(&rt).unwrap();

    ctx.with(|ctx| {
        let global = ctx.globals();
        global
            .set(
                "print",
                Function::new(ctx.clone(), print)
                    .unwrap()
                    .with_name("print")
                    .unwrap(),
            )
            .unwrap();

        println!("import script module");
        let _ = ctx
            .clone()
            .compile(
                "test",
                r#"
print(`f(2, 4) = 12`);
"#,
            )
            .unwrap();
    });
}
