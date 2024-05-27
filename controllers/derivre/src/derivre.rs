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

fn main() {
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
    //
    //
    //

}
