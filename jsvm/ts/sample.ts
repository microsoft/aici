import { fixed, gen_text, start } from "./aici";

async function main() {
    console.log("start")
  await fixed("2 + 2 =");
  console.log("fixed done")
  await gen_text({ max_tokens: 5 });
}

start(main);
