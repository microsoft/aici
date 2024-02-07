const fs = require('fs');
const path = require('path');
const exclude = []
fs.readFileSync('.gitmodules', 'utf8').replace(/^\s*path = (.*)$/gm, (_, path) => {
    exclude.push(path + "/")
})

const links = {}
const files = {}
for (let file of process.argv.slice(2)) {
    if (exclude.some(path => file.startsWith(path))) {
        continue
    }
    const content = fs.readFileSync(file, 'utf8')
    files[file] = content
    file = path.resolve(file)
    content.replace(/^#+ (.*)$/gm, (_, title) => {
        const anchor = "#" + title.toLowerCase().replace(/[^a-z0-9 \-]+/g, '').replace(/ /g, '-')
        links[file + anchor] = true
    })
}

let numerr = 0
let numlinks = 0
let numanchors = 0
let numhttp = 0

for (const [filename, content] of Object.entries(files)) {
    let lineNo = 0
    for (const line of content.split("\n")) {
        lineNo++
        line.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, (_, title, link) => {
            if (link.startsWith("https://") || link.startsWith("http://") || link.startsWith("mailto:")) {
                // console.log(link)
                numhttp++
                return
            }
            numlinks++
            if (link.startsWith("#")) {
                link = filename + link
            }
            // split anchor
            let [linkfile, anchor] = link.split("#")
            linkfile = path.resolve(path.dirname(filename), linkfile)

            if (!fs.existsSync(linkfile)) {
                numerr++
                console.log(`${filename}:${lineNo}: Broken link '${title}': ${link}`)
                return
            }

            if (anchor) {
                numanchors++
                if (!links[linkfile + "#" + anchor]) {
                    numerr++
                    console.log(`${filename}:${lineNo}: Broken link to anchor '${title}': ${link}`)
                } else {
                    // console.log(`${filename}:${lineNo}: Found link to anchor '${title}': ${link}`)
                }
            }
        })
    }
}

if (numerr > 0) {
    console.log(`Found ${numerr} broken links`)
    process.exit(1)
} else {
    console.log(`Exclude: ${exclude.join(", ")}`)
    console.log(`Checked ${numlinks} links (incl. ${numanchors} anchors). Skipped ${numhttp} http links.`)
}