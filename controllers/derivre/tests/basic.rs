use derivre::RegexVec;

fn check_is_match(rx: &mut RegexVec, s: &str, exp: bool) {
    if rx.is_match(s) == exp {
    } else {
        panic!(
            "error for: {:?}; expected {}",
            s,
            if exp { "match" } else { "no match" }
        );
    }
}

fn match_(rx: &mut RegexVec, s: &str) {
    check_is_match(rx, s, true);
}

fn match_many(rx: &mut RegexVec, ss: &[&str]) {
    for s in ss {
        match_(rx, s);
    }
}

fn no_match(rx: &mut RegexVec, s: &str) {
    check_is_match(rx, s, false);
}

fn no_match_many(rx: &mut RegexVec, ss: &[&str]) {
    for s in ss {
        no_match(rx, s);
    }
}

#[test]
fn test_basic() {
    let mut rx = RegexVec::new_single("a[bc](de|fg)").unwrap();
    println!("{:?}", rx);
    no_match(&mut rx, "abd");
    match_(&mut rx, "abde");

    no_match(&mut rx, "abdea");
    println!("{:?}", rx);

    let mut rx = RegexVec::new_single("a[bc]*(de|fg)*x").unwrap();

    no_match_many(&mut rx, &["", "a", "b", "axb"]);
    match_many(&mut rx, &["ax", "abdex", "abcbcbcbcdex", "adefgdefgx"]);
    println!("{:?}", rx);

    let mut rx = RegexVec::new_single("(A|foo)*").unwrap();
    match_many(
        &mut rx,
        &["", "A", "foo", "Afoo", "fooA", "foofoo", "AfooA", "Afoofoo"],
    );

    let mut rx = RegexVec::new_single("[abcquv][abdquv]").unwrap();
    match_many(
        &mut rx,
        &["aa", "ab", "ba", "ca", "cd", "ad", "aq", "qa", "qd"],
    );
    no_match_many(&mut rx, &["cc", "dd", "ac", "ac", "bc"]);

    println!("{:?}", rx);
}

#[test]
fn test_unicode() {
    let mut rx = RegexVec::new_single("źółw").unwrap();
    println!("{:?}", rx);
    no_match(&mut rx, "zolw");
    match_(&mut rx, "źółw");
    no_match(&mut rx, "źół");
    println!("{:?}", rx);

    let mut rx = RegexVec::new_single("[źó]łw").unwrap();
    match_(&mut rx, "ółw");
    match_(&mut rx, "źłw");
    no_match(&mut rx, "źzłw");

    let mut rx = RegexVec::new_single("x[©ª«]y").unwrap();
    match_many(&mut rx, &["x©y", "xªy", "x«y"]);
    no_match_many(&mut rx, &["x®y", "x¶y", "x°y", "x¥y"]);

    let mut rx = RegexVec::new_single("x[ab«\u{07ff}\u{0800}]y").unwrap();
    match_many(&mut rx, &["xay", "xby", "x«y", "x\u{07ff}y", "x\u{0800}y"]);
    no_match_many(&mut rx, &["xcy", "xªy", "x\u{07fe}y", "x\u{0801}y"]);

    let mut rx = RegexVec::new_single("x[ab«\u{07ff}-\u{0801}]y").unwrap();
    match_many(
        &mut rx,
        &[
            "xay",
            "xby",
            "x«y",
            "x\u{07ff}y",
            "x\u{0800}y",
            "x\u{0801}y",
        ],
    );
    no_match_many(&mut rx, &["xcy", "xªy", "x\u{07fe}y", "x\u{0802}y"]);
}

