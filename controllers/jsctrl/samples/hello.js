function fib(n) {
    if (n <= 1) return n
    return fib(n - 1) + fib(n - 2)
}

async function main() {
    await $`Ultimate answer is to the life, universe and everything is `
    await gen({ regex: /\d\d/ })
    console.log(`fib(10) = ${fib(30)}`)
    await gen({ regex: / [A-Z].*/ })
}

start(main)
