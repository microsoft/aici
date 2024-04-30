import { Label, setLogLevel } from "aici"

const colors = [
    // "red",
    // "green",
    "blue",
    // "violet",
    // "white",
    // "black",
]

function gencolors(k) {
    let s = []

    for (let i = 0; i < k; ++i) {
        s.push(colors[Math.floor(Math.random() * colors.length)])
    }

    return s.join(", ")
}

async function countYellow() {
    setLogLevel(10)
    const q = `Does the color yellow appear in the following list of colors?`
    await $`<|user|>\n${q}\n`
    const l = new Label()
    for (let i = 0; i < 100; ++i) {
        const hasYellow = Math.random() < 0.5
        const A = (10 + Math.random() * 100) | 0
        const B = (10 + Math.random() * 130) | 0
        const text = gencolors(A) + (hasYellow ? ", yellow, " : ", blue, ") + gencolors(B)
        await l.fixedAfter(`${text}<|end|>\n<|assistant|>`)
        const r = await gen({ maxTokens: 10, regex: /(Yes|No)/ })
        console.log(q)
        console.log(text)
        console.log(hasYellow ? "Yes" : "No", r)
        assert(r === "Yes" || r === "No")
        assert(r === (hasYellow ? "Yes" : "No"))
        await $`\n`
    }
}

start(countYellow)
