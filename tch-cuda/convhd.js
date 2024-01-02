const fs = require('fs');

const args = process.argv.slice(2);

let all_ext_c = "";
let all_rust = "";

const tpmap = {
    "torch::Tensor&": ["tensor", "*mut C_tensor"],
    "const torch::Tensor&": ["tensor", "*const C_tensor"],
    "const c10::optional<torch::Tensor>&": ["tensor", "*const C_tensor"],
};

const r_tpmap = {
    "float": "f32",
    "int": "i32",
    "bool": "bool",
}

let left = fs.readFileSync(args[0], 'utf-8').replace(/^(\S+) (\w+)\(([^)]*)\);/mg, (_, rettp, fname, args) => {
    let skip = false;

    if (rettp != "void")
        skip = true;

    if (args.indexOf("std::map<") >= 0) {
        return ""
    }

    let ext_c = `char* ${fname}_C(\n`;
    let rust = `fn ${fname}_C(\n`;
    let ext_c_inner = `${fname}(`;

    args.split(/,\s+/).forEach((x, i, arr) => {
        const m = /^(.*)\s+(\w+)$/.exec(x.trim());
        let tp = m[1];
        let aname = m[2];

        if (tp == "torch::Tensor")
            skip = true;


        const tp0 = tpmap[tp];

        let rtp = tp0?.[1] ?? r_tpmap[tp] ?? `C__${tp}`;
        let ctp = tp0?.[0] ?? tp;

        ext_c += `    ${ctp} ${aname}`;

        if (tp0) {
            ext_c_inner += `*${aname}`
        } else {
            ext_c_inner += `${aname}`
        }

        rust += `    ${aname}: ${rtp}`;

        if (i < arr.length - 1) {
            ext_c += ",\n";
            ext_c_inner += ",\n";
            rust += ",\n";
        }
    })

    ext_c += ") {\nPROTECT(" + ext_c_inner + "));\n}\n\n";
    rust += ") -> *mut libc::c_char;\n\n";

    if (!skip) {
        all_ext_c += ext_c;
        all_rust += rust;
    }

    return "";
});

left = left.replace(/^\s*#.*/mg, "").trim();
if (left) {
    console.log("left", left);
} else {
    console.log(all_ext_c);
    console.log(all_rust);
}
