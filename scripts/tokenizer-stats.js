const fs = require('fs');

const LONG_ID = 200;

function stats(fn) {
  console.log(fn)
  const tokenizer = JSON.parse(fs.readFileSync(fn, 'utf8'))

  const numbylen = {}
  let max_id = 0

  for (const [str, id] of Object.entries(tokenizer.model.vocab)) {
    const len = Math.floor(str.length / 10)
    if (!numbylen[len]) {
      numbylen[len] = 0
    }
    numbylen[len] += 1
    max_id = Math.max(max_id, id)
    if (str.length > LONG_ID) {
      console.log("Long token: ", id, JSON.stringify(str))
    }
  }

  console.log("Max ID: ", max_id)
  console.log("Length distribution:")
  for (const [len0, num] of Object.entries(numbylen)) {
    const len = parseInt(len0)
    console.log(`   ${len * 10}-${(len + 1) * 10 - 1}`, num)
  }
}

for (let i = 2; i < process.argv.length; i++) {
  stats(process.argv[i])
}