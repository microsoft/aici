const fs = require("fs")

const nevers = [0, 1, 2, 16, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 32000]

const MIN_DELTA = 0
const MAX_DELTA = MIN_DELTA + 2

function diff(a, b) {
    let i = 0
    let j = 0
    let delta = 0
    while (i < a.length || j < b.length) {
        if (i < a.length && j < b.length) {
            if (a[i] == b[j]) {
                i++;
                j++;
            } else if (a[i] < b[j]) {
                delta++;
                i++;
            } else {
                delta++;
                j++;
            }
        } else if (i < a.length) {
            delta++;
            i++;
        } else {
            delta++;
            j++;
        }

        if (delta > MAX_DELTA) {
            return delta;
        }
    }
    return delta;
}

const buckets = []
let no_bucket_size = 0

fs.readFileSync("tmp/tokens.txt", "utf8").split("\n").forEach((line, i) => {
    const m = /^(\d+) ==> (true|false) (.*)/.exec(line);
    if (!m) return
    const tokid = +m[1];
    let elts = Array.from(JSON.parse(m[3]));
    const neg = m[2] == "true";
    const isAllowed = (e) => {
        if (neg) return !elts.includes(e);
        return elts.includes(e);
    }

    const nev = nevers.find(e => isAllowed(e));
    if (nev) {
        console.log(tokid, "N", nev);
    }

    const empty = elts.length == 0 && !neg;
    if (empty) {
        //console.log(tokid, "E");
    } else {
        if (!neg) {
            console.log(tokid, "A", elts.length);
        } else {
            let existing = false
            elts = elts.filter(e => !nevers.includes(e));
            for (const b of buckets) {
                if (diff(elts, b) <= MIN_DELTA) {
                    existing = true;
                    break;
                }
            }
            if (!existing) {
                buckets.push(elts);
            }
            no_bucket_size += elts.length;
            console.log(tokid, "F", elts.length, buckets.length);
        }
    }
})

console.log(buckets.reduce((a, b) => a + b.length, 0), no_bucket_size)
