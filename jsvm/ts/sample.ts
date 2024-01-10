import { getVar, setVar } from "_aici";
import {
  $,
  Label,
  assert,
  checkVars,
  fork,
  gen,
  getPrompt,
  start,
  waitVars,
} from "./aici";

async function main() {
  await $`2 + 2 =`;
  await gen({ maxTokens: 5 });
}

async function test_sample() {
  // initialization code
  console.log("I'm going in the logs!");
  // ... more initialization code, it has long time limit
  const _prompt = await getPrompt();
  // here we're out of initialization code - the time limits are tight

  // This appends the exact string to the output; similar to adding it to prompt
  await $`The word 'hello' in French is`;

  // generate text (tokens) matching the regex
  const french = await gen({ regex: / "[^"]+"/, maxTokens: 5 });
  // set a shared variable (they are returned as JSON and are useful with aici.fork())
  setVar("french", french);

  await $` and in German`;
  // shorthand for the above
  await gen({ regex: / "[^"]+"/, storeVar: "german" });

  await $`\nFive`;
  // generates one of the strings
  await gen({ options: [" pounds", " euros", " dollars"] });
}

async function test_backtrack_one() {
  await $`3+`;
  const l = new Label();
  await $`2`;
  const x = await gen({ regex: /=\d\d?\./, storeVar: "x", maxTokens: 5 });
  console.log("X", x);
  await l.fixedAfter("4");
  const y = await gen({ regex: /=\d\d?\./, storeVar: "y", maxTokens: 5 });
  console.log("Y", y);
  checkVars({ x: "=5.", y: "=7." });
}

async function test_fork() {
  await $`The word 'hello' in`;
  const id = await fork(3);
  if (id === 0) {
    const [french, german] = await waitVars("french", "german");
    await $`${french} is the same as ${german}.`;
    await gen({ maxTokens: 5 });
    checkVars({ german: ' "hallo"', french: ' "bonjour"' });
  } else if (id === 1) {
    await $` German is`;
    await gen({ regex: / "[^"\.]+"/, storeVar: "german", maxTokens: 5 });
  } else if (id === 2) {
    await $` French is`;
    await gen({ regex: / "[^"\.]+"/, storeVar: "french", maxTokens: 5 });
  }
}

async function test_backtrack_lang() {
  await $`The word 'hello' in`;
  const l = new Label();
  await l.fixedAfter(` French is`);
  await gen({ regex: / "[^"\.]+"/, storeVar: "french", maxTokens: 5 });
  await l.fixedAfter(` German is`);
  await gen({ regex: / "[^"\.]+"/, storeVar: "german", maxTokens: 5 });
  checkVars({ french: ' "bonjour"', german: ' "hallo"' });
}

async function test_main() {
  console.log("start");
  console.log(getVar("test"));
  setVar("test", "hello");
  const v = getVar("test");
  console.log(typeof v);
  const prompt = await getPrompt();
  console.log(prompt);
  await $`The word 'hello' in French is`;
  await gen({ storeVar: "french", maxTokens: 5 });
  await $`\nIn German it translates to`;
  await gen({ regex: / "[^"]+"/, storeVar: "german" });
  await $`\nFive`;
  await gen({
    storeVar: "five",
    options: [" pounds", " euros"],
  });
  await $` is worth about $`;
  await gen({ regex: /\d+\.\d/, storeVar: "dollars" });
  checkVars({
    test: "hello",
    french: " 'bonjour'.",
    german: ' "guten Tag"',
    five: " pounds",
    dollars: "7.5",
  });
}

async function test_drugs() {
  const drug_syn =
    "\nUse <drug>Drug Name</drug> syntax for any drug name, for example <drug>Advil</drug>.\n\n";
  let notes =
    "The patient should take some tylenol in the evening and aspirin in the morning. Exercise is highly recommended. Get lots of sleep.\n";
  notes = "Start doctor note:\n" + notes + "\nEnd doctor note.\n";

  await $`[INST] `;
  const start = new Label();

  function inst(s: string) {
    return s + drug_syn + notes + " [/INST]\n";
  }

  await $`${inst(
    "List specific drug names in the following doctor's notes."
  )}\n1. <drug>`;
  const s = await gen({ maxTokens: 30 });

  const drugs: string[] = [];
  ("<drug>" + s).replace(/<drug>([^<]*)<\/drug>/g, (_, d) => {
    drugs.push(d);
    return "";
  });
  console.log("drugs", drugs);

  await start.fixedAfter(
    inst(
      "Make a list of each drug along with time to take it, based on the following doctor's notes."
    ) + `Take <drug>`
  );
  const pos = new Label();
  await gen({ options: drugs.map((d) => d + "</drug>") });

  for (let i = 0; i < 5; i++) {
    const fragment = await gen({ maxTokens: 20, stopAt: "<drug>" });
    console.log(fragment);
    if (fragment.includes("<drug>")) {
      assert(fragment.endsWith("<drug>"));
      await gen({ options: drugs.map((d) => d + "</drug>") });
    } else {
      break;
    }
  }

  setVar("times", pos.textSince());

  checkVars({
    times:
      "<drug>Tylenol</drug> in the evening.\n" +
      "Take <drug>Aspirin</drug> in the morning.\n" +
      "Exercise is highly recommended.\nGet lots of sleep.",
  });
}

start(test_sample);
