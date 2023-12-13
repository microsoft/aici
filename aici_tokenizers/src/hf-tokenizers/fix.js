const fs = require('fs');

const args = process.argv.slice(2);
const tok = JSON.parse(fs.readFileSync(args[0], 'utf8'));
const r = {}
for (const t of tok.added_tokens) {
    r[t.content] = t.id
}
console.log(JSON.stringify(r, null, 1))

