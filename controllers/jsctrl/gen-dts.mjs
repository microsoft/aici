import * as fs from 'fs';

function gen() {
    const ts = "./ts/"
    const native = fs.readFileSync(ts + '/native.d.ts', 'utf8')
    let aici = fs.readFileSync(ts + '/dist/aici.d.ts', 'utf8')
    aici = aici.replace(/<reference path=".*/g, "")
    aici = aici.replace(/ declare /g, " ")
    aici = aici.replace(/^declare /mg, "")
    aici = "declare module 'aici' {\n" + aici + "\n}\n"
    aici = native + aici
    fs.writeFileSync("./samples/aici-types.d.ts", aici)

    let jssrc = fs.readFileSync("../../py/pyaici/jssrc.py", 'utf8')
    jssrc = jssrc.replace(/(aici_types_d_t = r""")[^]*?"""/g, (_, pre) => pre + aici + '"""')

    const tsconfig = fs.readFileSync("./samples/tsconfig.json", "utf-8")
    jssrc = jssrc.replace(/(tsconfig_json = r""")[^]*?"""/g, (_, pre) => pre + tsconfig + '"""')

    const hello = fs.readFileSync("./samples/hello.js", "utf-8")
    jssrc = jssrc.replace(/(hello_js = r""")[^]*?"""/g, (_, pre) => pre + hello + '"""')

    fs.writeFileSync("../../py/pyaici/jssrc.py", jssrc)
}

gen()