// Doesn't seem to work too well...

import { Label } from "aici"

async function jsonString() {
    await gen({ 
        maxTokens: 50,
        regex: /"(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)+"/ 
    })
}

async function jsonInt() {
    await gen({ regex: /\d+/ })
}

/**
 * @param {string} name 
 */
async function jsonField(name) {
    await $`    "${name}": `
}

async function cityList() {
    const start = new Label()
    await $`[`
    const maxNodes = 3;
    for (let i = 0; i < maxNodes; i++) {
        await $`{\n`
        await jsonField("name")
        await jsonString()
        await $`,\n`
        await jsonField("population")
        await jsonInt()
        await $`,\n`
        await jsonField("url")
        await jsonString()
        await $`\n`
        const nextChar = await gen({ options: ['},\n', ']'] })
        if (nextChar === ']') {
            break
        }
    }
    console.log(start.textSince())
}

async function main() {
    await $`Here is JSON objects for five European cities:\n`
    await cityList()
}

start(main)
