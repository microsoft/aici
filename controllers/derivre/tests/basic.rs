use derivre::{NextByte, Regex};

fn check_is_match(rx: &mut Regex, s: &str, exp: bool) {
    if rx.is_match(s) == exp {
    } else {
        panic!(
            "error for: {:?}; expected {}",
            s,
            if exp { "match" } else { "no match" }
        );
    }
}

fn match_(rx: &mut Regex, s: &str) {
    check_is_match(rx, s, true);
}

fn match_many(rx: &mut Regex, ss: &[&str]) {
    for s in ss {
        match_(rx, s);
    }
}

fn no_match(rx: &mut Regex, s: &str) {
    check_is_match(rx, s, false);
}

fn no_match_many(rx: &mut Regex, ss: &[&str]) {
    for s in ss {
        no_match(rx, s);
    }
}

fn look(rx: &mut Regex, s: &str, exp: Option<usize>) {
    let res = rx.lookahead_len(s);
    if res == exp {
    } else {
        panic!(
            "lookahead len error for: {:?}; expected {:?}, got {:?}",
            s, exp, res
        )
    }
}

#[test]
fn test_basic() {
    let mut rx = Regex::new("a[bc](de|fg)").unwrap();
    println!("{:?}", rx);
    no_match(&mut rx, "abd");
    match_(&mut rx, "abde");

    no_match(&mut rx, "abdea");
    println!("{:?}", rx);

    let mut rx = Regex::new("a[bc]*(de|fg)*x").unwrap();

    no_match_many(&mut rx, &["", "a", "b", "axb"]);
    match_many(&mut rx, &["ax", "abdex", "abcbcbcbcdex", "adefgdefgx"]);
    println!("{:?}", rx);

    let mut rx = Regex::new("(A|foo)*").unwrap();
    match_many(
        &mut rx,
        &["", "A", "foo", "Afoo", "fooA", "foofoo", "AfooA", "Afoofoo"],
    );

    let mut rx = Regex::new("[abcquv][abdquv]").unwrap();
    match_many(
        &mut rx,
        &["aa", "ab", "ba", "ca", "cd", "ad", "aq", "qa", "qd"],
    );
    no_match_many(&mut rx, &["cc", "dd", "ac", "ac", "bc"]);

    println!("{:?}", rx);

    let mut rx = Regex::new("ab{3,5}c").unwrap();
    match_many(&mut rx, &["abbbc", "abbbbc", "abbbbbc"]);
    no_match_many(
        &mut rx,
        &["", "ab", "abc", "abbc", "abbb", "abbbx", "abbbbbbc"],
    );

    let mut rx = Regex::new("x*A[0-9]{5}").unwrap();
    match_many(&mut rx, &["A12345", "xxxxxA12345", "xA12345"]);
    no_match_many(&mut rx, &["A1234", "xxxxxA123456", "xA123457"]);
}

#[test]
fn test_unicode() {
    let mut rx = Regex::new("źółw").unwrap();
    println!("{:?}", rx);
    no_match(&mut rx, "zolw");
    match_(&mut rx, "źółw");
    no_match(&mut rx, "źół");
    println!("{:?}", rx);

    let mut rx = Regex::new("[źó]łw").unwrap();
    match_(&mut rx, "ółw");
    match_(&mut rx, "źłw");
    no_match(&mut rx, "źzłw");

    let mut rx = Regex::new("x[©ª«]y").unwrap();
    match_many(&mut rx, &["x©y", "xªy", "x«y"]);
    no_match_many(&mut rx, &["x®y", "x¶y", "x°y", "x¥y"]);

    let mut rx = Regex::new("x[ab«\u{07ff}\u{0800}]y").unwrap();
    match_many(&mut rx, &["xay", "xby", "x«y", "x\u{07ff}y", "x\u{0800}y"]);
    no_match_many(&mut rx, &["xcy", "xªy", "x\u{07fe}y", "x\u{0801}y"]);

    let mut rx = Regex::new("x[ab«\u{07ff}-\u{0801}]y").unwrap();
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

    let mut rx = Regex::new(".").unwrap();
    no_match(&mut rx, "\n");
    match_many(&mut rx, &["a", "1", " ", "\r"]);

    let mut rx = Regex::new("a.*b").unwrap();
    match_many(&mut rx, &["ab", "a123b", "a \r\t123b"]);
    no_match_many(&mut rx, &["a", "a\nb", "a1\n2b"]);
}

#[test]
fn test_lookaround() {
    let mut rx = Regex::new("[ab]*(?P<stop>xx)").unwrap();
    match_(&mut rx, "axx");
    look(&mut rx, "axx", Some(2));
    look(&mut rx, "ax", None);

    let mut rx = Regex::new("[ab]*(?P<stop>x*y)").unwrap();
    look(&mut rx, "axy", Some(2));
    look(&mut rx, "ay", Some(1));
    look(&mut rx, "axxy", Some(3));
    look(&mut rx, "aaaxxy", Some(3));
    look(&mut rx, "abaxxy", Some(3));
    no_match_many(&mut rx, &["ax", "bx", "aaayy", "axb", "axyxx"]);

    let mut rx = Regex::new("[abx]*(?P<stop>[xq]*y)").unwrap();
    look(&mut rx, "axxxxxxxy", Some(1));
    look(&mut rx, "axxxxxxxqy", Some(2));
    look(&mut rx, "axxxxxxxqqqy", Some(4));

    let mut rx = Regex::new("(f|foob)(?P<stop>o*y)").unwrap();
    look(&mut rx, "fooby", Some(1));
    look(&mut rx, "fooy", Some(3));
    look(&mut rx, "fy", Some(1));
}

#[test]
fn utf8_dfa() {
    let parser = regex_syntax::ParserBuilder::new()
        .unicode(false)
        .utf8(false)
        .ignore_whitespace(true)
        .build();

    let utf8_rx = r#"
   ( [\x00-\x7F]                        # ASCII
   | [\xC2-\xDF][\x80-\xBF]             # non-overlong 2-byte
   |  \xE0[\xA0-\xBF][\x80-\xBF]        # excluding overlongs
   | [\xE1-\xEC\xEE\xEF][\x80-\xBF]{2}  # straight 3-byte
   |  \xED[\x80-\x9F][\x80-\xBF]        # excluding surrogates
   |  \xF0[\x90-\xBF][\x80-\xBF]{2}     # planes 1-3
   | [\xF1-\xF3][\x80-\xBF]{3}          # planes 4-15
   |  \xF4[\x80-\x8F][\x80-\xBF]{2}     # plane 16
   )*
   "#;

    let mut rx = Regex::new_with_parser(parser, utf8_rx).unwrap();
    println!("UTF8 {:?}", rx);
    //match_many(&mut rx, &["a", "ą", "ę", "ó", "≈ø¬", "\u{1f600}"]);
    println!("UTF8 {:?}", rx);
    let compiled = rx.dfa();
    println!("UTF8 {:?}", rx);
    println!("mapping ({}) {:?}", rx.alpha().len(), &compiled[0..256]);
    println!("states {:?}", &compiled[256..]);
    println!("initial {:?}", rx.initial_state());
}

#[test]
fn utf8_restrictions() {
    let mut rx = Regex::new("(.|\n)*").unwrap();
    println!("{:?}", rx);
    match_many(&mut rx, &["", "a", "\n", "\n\n", "\x00", "\x7f"]);
    let s0 = rx.initial_state();
    assert!(rx.transition(s0, 0x80).is_dead());
    assert!(rx.transition(s0, 0xC0).is_dead());
    assert!(rx.transition(s0, 0xC1).is_dead());
    // more overlong:
    assert!(rx.transition_bytes(s0, &[0xE0, 0x80]).is_dead());
    assert!(rx.transition_bytes(s0, &[0xE0, 0x9F]).is_dead());
    assert!(rx.transition_bytes(s0, &[0xF0, 0x80]).is_dead());
    assert!(rx.transition_bytes(s0, &[0xF0, 0x8F]).is_dead());
    // surrogates:
    assert!(rx.transition_bytes(s0, &[0xED, 0xA0]).is_dead());
    assert!(rx.transition_bytes(s0, &[0xED, 0xAF]).is_dead());
    assert!(rx.transition_bytes(s0, &[0xED, 0xBF]).is_dead());
}

#[test]
fn trie() {
    let mut rx = Regex::new("(foo|far|bar|baz)").unwrap();
    match_many(&mut rx, &["foo", "far", "bar", "baz"]);
    no_match_many(&mut rx, &["fo", "fa", "b", "ba", "baa", "f", "faz"]);

    let mut rx = Regex::new("(foobarbazqux123|foobarbazqux124)").unwrap();
    match_many(&mut rx, &["foobarbazqux123", "foobarbazqux124"]);
    no_match_many(
        &mut rx,
        &["foobarbazqux12", "foobarbazqux125", "foobarbazqux12x"],
    );

    let mut rx = Regex::new("(1a|12a|123a|1234a|12345a|123456a)").unwrap();
    match_many(
        &mut rx,
        &["1a", "12a", "123a", "1234a", "12345a", "123456a"],
    );
    no_match_many(
        &mut rx,
        &["1234567a", "123456", "12345", "1234", "123", "12", "1"],
    );
}

#[test]
fn unicode_case() {
    let mut rx = Regex::new("(?i)Żółw").unwrap();
    match_many(&mut rx, &["Żółw", "żółw", "ŻÓŁW", "żóŁw"]);
    no_match_many(&mut rx, &["zółw"]);

    let mut rx = Regex::new("Żółw").unwrap();
    match_(&mut rx, "Żółw");
    no_match_many(&mut rx, &["żółw", "ŻÓŁW", "żóŁw"]);
}

fn validate_next_byte(rx: &mut Regex, data: Vec<(NextByte, u8)>) {
    let mut s = rx.initial_state();
    for (exp, b) in data {
        let nb = rx.next_byte(s);
        if nb != exp {
            panic!("expected {:?}, got {:?}", exp, nb);
        }
        if nb == NextByte::ForcedEOI {
            assert!(rx.is_accepting(s));
        } else if nb == NextByte::Dead {
            assert!(s.is_dead());
        }
        s = rx.transition(s, b);
        if nb == NextByte::ForcedEOI {
            assert!(s.is_dead());
            assert!(rx.next_byte(s) == NextByte::Dead);
        }
    }
}

#[test]
fn next_byte() {
    let mut rx = Regex::new("a[bc]*dx").unwrap();
    validate_next_byte(
        &mut rx,
        vec![
            (NextByte::ForcedByte(b'a'), b'a'),
            (NextByte::SomeBytes, b'b'),
            (NextByte::SomeBytes, b'd'),
            (NextByte::ForcedByte(b'x'), b'x'),
            (NextByte::ForcedEOI, b'x'),
        ],
    );

    rx = Regex::new("abdx|aBDy").unwrap();
    validate_next_byte(
        &mut rx,
        vec![
            (NextByte::ForcedByte(b'a'), b'a'),
            (NextByte::SomeBytes, b'B'),
            (NextByte::ForcedByte(b'D'), b'D'),
        ],
    );

    rx = Regex::new("foo|bar").unwrap();
    validate_next_byte(
        &mut rx,
        vec![
            (NextByte::SomeBytes, b'f'),
            (NextByte::ForcedByte(b'o'), b'o'),
            (NextByte::ForcedByte(b'o'), b'o'),
            (NextByte::ForcedEOI, b'X'),
        ],
    );
}
