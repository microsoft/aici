import { fixed, gen_text, start } from "./aici";
async function main() {
    await fixed("2 + 2 =");
    await gen_text({ max_tokens: 5 });
}
start(main);
