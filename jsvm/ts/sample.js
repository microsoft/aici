import { setVar } from "_aici";
import { $, gen, getPrompt, start } from "./aici";
async function main() {
    await $ `2 + 2 =`;
    await gen({ maxTokens: 5 });
}
async function test_sample() {
    // initialization code
    console.log("I'm going in the logs!");
    // ... more initialization code, it has long time limit
    const _prompt = await getPrompt();
    // here we're out of initialization code - the time limits are tight
    // This appends the exact string to the output; similar to adding it to prompt
    await $ `The word 'hello' in French is`;
    // generate text (tokens) matching the regex
    const french = await gen({ regex: / "[^"]+"/, maxTokens: 5 });
    // set a shared variable (they are returned as JSON and are useful with aici.fork())
    setVar("french", french);
    await $ ` and in German`;
    // shorthand for the above
    await gen({ regex: / "[^"]+"/, storeVar: "german" });
    await $ `\nFive`;
    // generates one of the strings
    await gen({ options: [" pounds", " euros", " dollars"] });
}
start(test_sample);
