const child_process = require("child_process")
const fs = require("fs")

const sysroot = child_process.execSync("rustc --print sysroot").toString().trim()

function main(sname, filter) {
    if (!filter) {
        console.error("please pass filter arg")
        return
    }

    const sections = {}
    const files = []
    let idx = 0
    for (const sect of fs.readFileSync(sname, "utf8").split("\n\n")) {
        idx++
        let sectId = "sect" + idx
        let m = /^\t\.type\t(.*),@/m.exec(sect)
        if (m) {
            sectId = m[1]
        }

        let outp = ""
        for (const line of sect.split("\n")) {
            if (line.startsWith(".Ltmp") || line.startsWith("\t.cfi_"))
                continue
            if (line.startsWith("\t.file\t")) {
                m = /(\d+)\s+"([^"]+)"\s+"([^"]+)"/.exec(line)
                if (!m) {
                    // console.error("Bad file line", line)
                } else {
                    const folder = m[2].replace(/^\/rustc\/[^/]+/, sysroot + "/lib/rustlib/src/rust")
                    files[+m[1]] = folder + "/" + m[3]
                }
                continue
            }
            outp += line + "\n"
        }

        sections[sectId] = outp
    }

    const keys = Object.keys(sections).filter(k => k.includes(filter))
    if (keys.length > 1) {
        const max = 50
        console.error("Multiple sections found for filter", filter, keys.slice(0, max).join("\n"))
        if (keys.length > max) {
            console.error("...")
        }
        return
    }
    if (keys.length === 0) {
        console.error("No sections found for filter", filter)
        return
    }

    const filecontent = []

    function fileLines(id) {
        if (filecontent[id]) {
            return filecontent[id]
        }
        const lines = fs.readFileSync(files[id], "utf8").split("\n")
        filecontent[id] = lines
        return lines
    }

    let outp = ""
    const labels = {}
    for (let line of sections[keys[0]].split("\n")) {
        if (line.startsWith("\t.loc\t")) {
            const m = /\t.loc\t(\d+)\s+(\d+)\s+(\d+)/.exec(line)
            const lineno = +m[2]
            const lines = fileLines(+m[1])
            const filename = files[+m[1]]
            let basename = filename.split("/").pop()
            if (filename.startsWith(sysroot))
                basename = "[lib]" + basename
            // outp += "// file://" + files[+m[1]] + "\n"
            if (lines[lineno - 1] !== undefined) {
                const tag = basename + ":" + lineno
                outp += "// " + tag.padEnd(40, " ") + lines[lineno - 1] + "\n"
            }
        } else {
            const m = /^(\.L\w+):/.exec(line)
            if (m) {
                labels[m[1]] = true
            }
            const words = line.split(/\s+/)
            if (words.some(w => labels[w])) {
                line += " // ===============================================> BACK"
            }
            outp += line + "\n"
        }
    }

    console.log("Section", keys[0], ":")
    console.log(outp)
}


const args = process.argv.slice(2)
main(...args)
