#!/usr/bin/env node

const fs = require('fs');

const CLASS_EXAMPLES = true;
const LONG_ID = 200;

const classes = {
  'EMPTY': /^$/,
  'PUNCT': /^[^\p{L}0-9]+$/u,
  'NUM': /^[0-9]+$/,
  'ALPHA': /^[\p{L}']+$/u,
  'SPECE_ALPHA': /^ [\p{L}']+$/u,
  'SPECE_NUM': /^ [0-9]+$/,
  'BYTE_FALLBACK': /^\<0x[0-9A-F][0-9A-F]\>$/,
  'PUNCT_ALPHA': /^[^\p{L}'0-9 ][\p{L}]+$/u,
  'BAD_UTF': /�/,
}


function charMap() {
  const res = {}
  let k = 0x100
  for (let byte = 0; byte <= 255; byte++) {
    const c = String.fromCharCode(byte)
    if (c.match(/[\!-\~\u00A1-\u00AC\u00AE-\u00FF]/)) {
      res[c] = byte
    } else {
      res[String.fromCharCode(k)] = byte
      k += 1
    }
  }
  return res
}

const char_map = charMap()

function tokenNameToBytes(tok_name) {
  const bytes = []
  for (const c of tok_name) {
    const code = char_map[c]
    if (code === undefined) {
      throw new Error(`missing char: ${c}`)
    }
    bytes.push(code)
  }
  return bytes
}

function tokenNameToString(tok_name) {
  return Buffer.from(tokenNameToBytes(tok_name)).toString('utf8')
}

function padNum(n) {
  return n.toString().padStart(6, ' ')
}


function stats(fn) {
  console.log(fn)
  const tokenizer = JSON.parse(fs.readFileSync(fn, 'utf8'))

  const addedTokens = {}
  for (const added of tokenizer.added_tokens) {
    addedTokens[added.content] = true
  }

  const numbylen = {}
  let max_id = 0
  const tokensByClass = {}

  const isFallback = tokenizer.model.vocab.hasOwnProperty("▁▁▁");

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
    const t = isFallback ? str.replace(/▁/g, " ") : tokenNameToString(str)
    const tclasses = []
    if (addedTokens.hasOwnProperty(str)) {
      tclasses.push('ADDED')
    } else {
      const t2 = t.replace(/\n/g, "\t")
      for (const [n, rx] of Object.entries(classes)) {
        if (t2.match(rx)) {
          tclasses.push(n)
        }
      }
    }
    if (tclasses.length == 0) {
      console.log("No class: ", id, JSON.stringify(t))
      tclasses.push('UNKNOWN')
    } else if (tclasses.length > 1 && !tclasses.includes('BAD_UTF') && !/^[ ']+$/.test(t)) {
      console.log("Multiple classes: ", id, JSON.stringify(t), tclasses)
    }

    for (const c of tclasses) {
      if (!tokensByClass[c]) {
        tokensByClass[c] = []
      }
      tokensByClass[c].push(t)
    }
  }

  console.log("Max ID: ", max_id)
  console.log("Length distribution:")
  for (const [len0, num] of Object.entries(numbylen)) {
    const len = parseInt(len0)
    console.log(padNum(num), `${len * 10}-${(len + 1) * 10 - 1}`)
  }
  console.log("Classes:")
  for (const [c, elts] of Object.entries(tokensByClass)) {
    console.log(padNum(elts.length), c, "   ", classes[c])
    if (CLASS_EXAMPLES) {
      console.log("     ", JSON.stringify(permute(elts).slice(0, 10)))
    }
  }
}

function permute(arr) {
  // randomly permute elts of array
  const res = arr.slice()
  for (let i = 0; i < arr.length; i++) {
    const j = Math.floor(Math.random() * arr.length)
    const tmp = res[i]
    res[i] = res[j]
    res[j] = tmp
  }
  return res
}

for (let i = 2; i < process.argv.length; i++) {
  stats(process.argv[i])
}