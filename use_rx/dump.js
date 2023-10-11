const fs = require("fs");

let d = fs
  .readFileSync(process.argv[2], "utf-8")
  .split(/\n/)
  .filter((s) => s.trim().startsWith("(data "))[0];
d = d
  .replace(/^[^"]*/, "")
  .replace(/\)$/, "")
  .replace(/\\([0-9a-f]{2}|.)/g, (_, a) => {
    if (a.length == 2) return "\\u00" + a;
    else if (a == "\\" || a == '"') return "\\" + a;
    else return a;
  });
//console.log(d.slice(34496 - 1, 34496 + 20));
fs.writeFileSync(
  process.argv[2] + ".bin",
  Buffer.from(JSON.parse(d), "latin1")
);
