use std::fmt::{Debug, Display};

const BYTESET_LEN: usize = 8;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ByteSet {
    mask: [u32; BYTESET_LEN],
}

impl Debug for ByteSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for i in 0u32..=256 {
            if i <= 0xff && self.contains(i as u8) {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                write!(f, "{}", i)?;
            }
        }
        write!(f, "]")
    }
}

pub fn byte_to_string(b: u8) -> String {
    if b >= 0x7f {
        format!("x{:02x}", b)
    } else {
        let b = b as char;
        match b {
            '_' | 'a'..='z' | 'A'..='Z' | '0'..='9' => format!("{}", b),
            _ => format!("{:?}", b as char),
        }
    }
}

impl Display for ByteSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut start = None;
        let mut first = true;
        for i in 0u32..=256 {
            if i <= 0xff && self.contains(i as u8) {
                if start.is_none() {
                    start = Some(i);
                }
            } else {
                if let Some(start) = start {
                    if !first {
                        write!(f, ";")?;
                    }
                    first = false;
                    write!(f, "{}", byte_to_string(start as u8))?;
                    if i - start > 1 {
                        write!(f, "-{}", byte_to_string((i - 1) as u8))?;
                    }
                }
                start = None;
            }
        }
        Ok(())
    }
}

impl ByteSet {
    pub fn new() -> Self {
        ByteSet {
            mask: [0; BYTESET_LEN],
        }
    }

    pub fn from_sum<'a>(elts: impl Iterator<Item = ByteSet>) -> Self {
        let mut r = ByteSet::new();
        for e in elts {
            r.add_set(&e);
        }
        r
    }

    pub fn add_set(&mut self, other: &ByteSet) {
        for i in 0..BYTESET_LEN {
            self.mask[i] |= other.mask[i];
        }
    }

    pub fn add(&mut self, byte: u8) {
        let idx = byte as usize / 32;
        let bit = byte as usize % 32;
        self.mask[idx] |= 1 << bit;
    }

    pub fn contains(&self, byte: u8) -> bool {
        let idx = byte as usize / 32;
        let bit = byte as usize % 32;
        self.mask[idx] & (1 << bit) != 0
    }

    pub fn from_range(start: u8, end: u8) -> Self {
        let mut r = ByteSet::new();
        // TODO optimize
        for b in start..=end {
            r.add(b);
        }
        r
    }

    pub fn num_bytes(&self) -> usize {
        let mut r = 0;
        for i in 0..BYTESET_LEN {
            r += self.mask[i].count_ones() as usize;
        }
        r
    }

    pub fn first_byte(&self) -> Option<u8> {
        for i in 0..BYTESET_LEN {
            let m = self.mask[i];
            if m != 0 {
                let bit = m.trailing_zeros() as usize;
                return Some((i * 32 + bit) as u8);
            }
        }
        None
    }

    pub fn single_byte(&self) -> Option<u8> {
        if self.num_bytes() != 1 {
            None
        } else {
            self.first_byte()
        }
    }
}
