import { Label, getTokens, setLogLevel } from "aici"

function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min)
}

async function countYellow() {
    setLogLevel(0)
    const q = `Tell me the value of x42?`
    await $`<|user|>\n${q}\n`
    const l = new Label()
    let numok = 0
    for (let i = 0; i < 20; ++i) {
        let text = ""
        let x42 = randomInt(10, 99)
        for (let i = 10; i < 300; ++i) {
            text += `The value of x${i} is ${i == 42 ? x42 : randomInt(10, 99)}.\n`
        }
        await l.fixedAfter(`${text}\nTell me x42.<|end|>\n<|assistant|>The value of x42 is `)
        const r = await gen({ maxTokens: 10, regex: /\d\d/ })
        // console.log(q)
        // console.log(text)
        console.log(getTokens().length, x42, r, r === x42.toString())
        if (r === x42.toString()) {
            numok++
        }
        // assert(r === x42.toString())
        await $`\n`
    }
    console.log("numok", numok)
}

start(countYellow)
