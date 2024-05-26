use derivre::RegexVec;


fn main() {
    let mut rx = RegexVec::new_single("a[bc](de|fg)").unwrap();
    println!("{:?}", rx);
    assert!(rx.is_match("abde"));
    println!("{:?}", rx);
}