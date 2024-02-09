const fs = require('fs');
const path = require('path');

const cargoToml = fs.readFileSync('Cargo.toml', 'utf8');
let m = /\[dependencies\]\n([^]*)/m.exec(cargoToml);
const deps = m[1].replace(/\n\[[^]*/, "").split('\n').map(line => line.replace(/=.*/, "").trim()).filter(line => line.length > 0);
const depset = new Set(deps);

// read all *.rs files under src/ recursively
const srcFiles = [];
function readDir(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const filePath = path.join(dir, file);
        if (fs.statSync(filePath).isDirectory()) {
            readDir(filePath);
        }
        else if (file.endsWith('.rs')) {
            const content = fs.readFileSync(filePath, 'utf8');
            for (const dep of depset) {
                const dn = dep.replace(/-/g, "_");
                if (content.includes(`${dn}::`)) {
                    depset.delete(dep);
                }
            }
            srcFiles.push(filePath);
        }
    }
}
readDir('src');
console.log('Unused dependencies:', depset);
