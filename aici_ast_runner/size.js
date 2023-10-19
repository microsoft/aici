const fs = require("fs");

// run "cmd" and capture the output
function run(cmd) {
  return new Promise((resolve, reject) => {
    require("child_process").exec(cmd, (error, stdout, stderr) => {
      if (error) {
        reject(error);
      } else {
        resolve(stdout.trim());
      }
    });
  });
}

function fmt(perc) {
  return perc.toFixed(2).padStart(6) + "%";
}

function gethd(o) {
  return (
    fmt(o.retained_size_percent) +
    " " +
    fmt(o.shallow_size_percent) +
    " " +
    o.name
  );
}

function cchildren(o) {
  const r = {};
  (o.children ?? []).forEach((c) => {
    r[gethd(c)] = cchildren(c);
  });
  return r;
}

async function main() {
  const o = JSON.parse(
    await run("twiggy dominators target/strip.wasm -f json")
  );
  o.root = cchildren({ name: "ROOT", children: o.items });
  delete o.items;
  fs.writeFileSync(
    "target/dominators.json",
    JSON.stringify(o, null, 2)
  );
}
main();
