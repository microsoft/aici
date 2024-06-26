use derivre::{RegexVec, SimpleVob};

#[test]
fn multi_matching_sample() {
    let mut rx = RegexVec::new_vec(&["if", "while", "[a-z]+"]).unwrap();
    // initial state with all patterns enabled
    let enabled = SimpleVob::from_slice(&[true, true, true]);
    let state0 = rx.initial_state(&enabled);
    let state1 = rx.transition_bytes(state0, b"while");
    let desc = rx.state_desc(state1);
    // first accepting pattern is #1 ("while")
    assert!(desc.lowest_accepting == Some(1));
    assert!(desc.accepting[0] == false);
    assert!(desc.accepting[1] == true);
    // but [a-z]+ is also matching
    assert!(desc.accepting[2] == true);
    // same as the three asserts above
    assert!(desc.accepting.to_bin_string().as_str() == "011");

    let state2 = rx.transition_bytes(state0, b"i");
    let desc = rx.state_desc(state2);
    // after we go through just 'i' from start, we can only match [a-z]+
    assert!(desc.lowest_accepting == Some(2));
    assert!(desc.accepting.to_bin_string().as_str() == "001");
    // however, the 'if' is still possible (but 'while' is not)
    assert!(desc.possible.to_bin_string().as_str() == "101");

    // here we disable 'if' pattern
    let enabled = SimpleVob::from_slice(&[false, true, true]);
    let state0 = rx.initial_state(&enabled);
    let state1 = rx.transition_bytes(state0, b"if");
    let desc = rx.state_desc(state1);
    // the string matches the identifier rule only
    assert!(desc.lowest_accepting == Some(2));
    assert!(desc.accepting.to_bin_string().as_str() == "001");
}
