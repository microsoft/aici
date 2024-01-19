async function main() {
    await $`Hello`
    await gen({ regex: / [A-Z]+/ })
}

start(main)
