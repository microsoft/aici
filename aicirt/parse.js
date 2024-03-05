/*

start!
time: 25.139657ms - 26.426091ms - 29.282678ms 5945 6521

 Performance counter stats for '/workspaces/aici/target/release/aicirt --tokenizer gpt4 --save-tokenizer tmp/foo.bin':

           2033.82 msec task-clock                #    1.000 CPUs utilized          
                 4      context-switches          #    1.967 /sec                   
                 0      cpu-migrations            #    0.000 /sec                   
             21331      page-faults               #   10.488 K/sec                  
        6698943270      cycles                    #    3.294 GHz                      (83.28%)
          12761596      stalled-cycles-frontend   #    0.19% frontend cycles idle     (83.28%)
          11202707      stalled-cycles-backend    #    0.17% backend cycles idle      (83.28%)
       26662989271      instructions              #    3.98  insn per cycle         
                                                  #    0.00  stalled cycles per insn  (83.31%)
        5501295234      branches                  #    2.705 G/sec                    (83.48%)
          12446413      branch-misses             #    0.23% of all branches          (83.36%)

       2.034238333 seconds time elapsed

       1.990138000 seconds user

*/

const fs = require('fs');
const keys = [
    "msec task-clock",
    "cycles",
    "instructions",
    "branches",
    "seconds time elapsed",
    "seconds user"
]

const k2 = ["min-ms", "avg-ms", "max-ms", "min-us", "med-us"]

const data = {}

fs.readFileSync('tmp/log.txt', 'utf8').split('\nstart!\n').forEach((sect) => {
    const m = /^time: ([\d.]+)ms - ([\d.]+)ms - ([\d.]+)ms (\d+) (\d+)$/m.exec(sect)
    if (!m) {
        console.log("Bad time line", sect)
        return
    }
    k2.forEach((k, i) => {
        if (!data[k]) data[k] = []
        data[k].push(+m[i + 1])
    })
    keys.forEach((k) => {
        if (!data[k]) data[k] = []
        const m = new RegExp("([\\d.]+)\\s+" + k).exec(sect)
        data[k].push(+m[1])
    })
})

function normalize(arr) {
    arr.sort()
    const med = arr[Math.floor(arr.length / 2)]
    for (let i = 0; i < arr.length; i++) {
        arr[i] /= med
    }
}

let maxlen = 0

for (const arr of Object.values(data)) {
    normalize(arr)
    if (arr.length > maxlen) maxlen = arr.length
}

function csv(arr) {
    return arr.map(s => JSON.stringify("" + s)).join(",")
}

let outp = csv(Object.keys(data))
for (let i = 0 ; i < maxlen; i++) {
    outp += "\n" + csv(Object.values(data).map(arr => arr[i]))
}

fs.writeFileSync('tmp/log.csv', outp)

