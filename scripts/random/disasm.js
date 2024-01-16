const { exec } = require("child_process");

let args = process.argv.slice(2);
if (args.length != 1 || !args[0].endsWith(".wasm")) {
    console.log("Usage: node scripts/disasm.js cache/<file>.wasm");
    console.log("Expects cache/<file>.elf to also exist");
    process.exit(1);
}

function run(cmd, cb) {
    exec(cmd, {
        encoding: "utf-8",
        maxBuffer: 128 * 1024 * 1024,
    }, (error, stdout, stderr) => {
        if (error) {
            console.log(`error: ${error.message}`)
            process.exit(1)
        }
        if (stderr) {
            console.log(`stderr: ${stderr}`);
            process.exit(1)
        }
        cb(stdout)
    });
}

run(`wasm-objdump -x ${args[0]} | rustfilt -h`, stdout => {
    const repl = {}
    stdout.split("\n").forEach(line => {
        // - func[6] sig=0 <alloc::raw_vec::RawVec<T,A>::reserve_for_push::h27d4ac8d729e40c6>
        const m = /^\s*- func\[(\d+)\] .* <(.*)>$/.exec(line);
        if (m) {
            repl[m[1]] = m[2];
        }
        return ""
    })

    run(`objdump -d ${args[0].replace(".wasm", ".elf")}`, stdout => {
        console.log(stdout
            .replace(
                /<[^<>]*(\+0x[a-f0-9]+)>/g,
                (_, addr) => addr)
            .replace(
                /wasm\[0\]::function\[(\d+)\]/g,
                (_, no) => {
                    return repl[no] || `wasm[0]::function[${no}]`
                }
            ))
    })
});