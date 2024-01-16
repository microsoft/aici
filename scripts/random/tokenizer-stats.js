const fs = require("fs");
const folder = "src/tokenizers/";

for (const fn of fs.readdirSync(folder)) {
  if (!fn.endsWith(".json")) continue;
  // if (fn != "gpt4.json") continue;
  console.log(fn);

  const obj = JSON.parse(fs.readFileSync(folder + fn, "utf-8"));

  const keys = [];

  for (const k of Object.keys(obj.binary)) {
    keys.push(Buffer.from(k, "hex").toString("utf-8"));
  }
  keys.push(...Object.keys(obj.text));
  keys.sort((a, b) => b.length - a.length);

  // console.log( keys.filter(k => /^\s*$/.test(k)).map(x => JSON.stringify(x)).join("\n") )
  // console.log(
  //   keys
  //     .filter((k) => /^\s*$/.test(k))
  //     .filter((k) => !/^[ \t\r\n]*$/.test(k) && k.length != 1)
  //     .map((x) => JSON.stringify(x))
  //     .join("\n")
  // );

  const nonws = keys.filter((k) => !/^\s*$/.test(k));

  console.log(
    nonws
      .filter((k) => / /.test(k))
      .filter((k) => k[0] != " ")
      .map((x) => JSON.stringify(x))
      .join("\n")
  );
}
