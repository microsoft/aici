use crate::ast::{
    byteset_contains, byteset_set, byteset_union, Expr, ExprFlags, ExprRef, ExprSet, ExprTag,
};

impl ExprSet {
    fn pay(&mut self, cost: usize) {
        self.cost += cost;
    }

    pub fn mk_byte(&mut self, b: u8) -> ExprRef {
        self.pay(1);
        self.mk(Expr::Byte(b))
    }

    pub fn mk_byte_set(&mut self, s: &[u32]) -> ExprRef {
        assert!(s.len() == self.alphabet_words);
        self.pay(self.alphabet_words);
        let mut num_set = 0;
        for x in s.iter() {
            num_set += x.count_ones();
        }
        if num_set == 0 {
            ExprRef::NO_MATCH
        } else if num_set == 1 {
            for i in 0..self.alphabet_size {
                if byteset_contains(s, i) {
                    return self.mk_byte(i as u8);
                }
            }
            unreachable!()
        } else {
            self.mk(Expr::ByteSet(s))
        }
    }

    pub fn mk_repeat(&mut self, e: ExprRef, min: u32, max: u32) -> ExprRef {
        self.pay(1);
        if e == ExprRef::NO_MATCH {
            if min == 0 {
                ExprRef::EMPTY_STRING
            } else {
                ExprRef::NO_MATCH
            }
        } else if e == ExprRef::EMPTY_STRING {
            ExprRef::EMPTY_STRING
        } else if min > max {
            panic!();
            // ExprRef::NO_MATCH
        } else if max == 0 {
            ExprRef::EMPTY_STRING
        } else if min == 1 && max == 1 {
            e
        } else {
            let min = if self.is_nullable(e) { 0 } else { min };
            let flags = ExprFlags::from_nullable(min == 0);
            self.mk(Expr::Repeat(flags, e, min, max))
        }
    }

    // pub fn mk_star(&mut self, e: ExprRef) -> ExprRef {
    //     self.mk_repeat(e, 0, u32::MAX)
    // }

    // pub fn mk_plus(&mut self, e: ExprRef) -> ExprRef {
    //     self.mk_repeat(e, 1, u32::MAX)
    // }

    fn flatten_tag(&self, exp_tag: ExprTag, args: Vec<ExprRef>) -> Vec<ExprRef> {
        let mut i = 0;
        while i < args.len() {
            let tag = self.get_tag(args[i]);
            if tag == exp_tag {
                // ok, we found tag, we can no longer return the original vector
                let mut res = args[0..i].to_vec();
                while i < args.len() {
                    let tag = self.get_tag(args[i]);
                    if tag != exp_tag {
                        res.push(args[i]);
                    } else {
                        res.extend_from_slice(self.get_args(args[i]));
                    }
                    i += 1;
                }
                return res;
            }
            i += 1;
        }
        args
    }

    // Complexity of mk_X(args) is O(n log n) where n = |flatten(X, args)|

    pub fn mk_or(&mut self, mut args: Vec<ExprRef>) -> ExprRef {
        // TODO deal with byte ranges
        args = self.flatten_tag(ExprTag::Or, args);
        self.pay(args.len());
        args.sort_unstable();
        let mut dp = 0;
        let mut prev = ExprRef::NO_MATCH;
        let mut nullable = false;
        let mut num_bytes = 0;
        let mut num_lookahead = 0;
        for idx in 0..args.len() {
            let arg = args[idx];
            if arg == prev || arg == ExprRef::NO_MATCH {
                continue;
            }
            if arg == ExprRef::ANY_STRING {
                return ExprRef::ANY_STRING;
            }
            match self.get(arg) {
                Expr::Byte(_) | Expr::ByteSet(_) => {
                    num_bytes += 1;
                }
                Expr::Lookahead(_, _, _) => {
                    num_lookahead += 1;
                }
                _ => {}
            }
            if !nullable && self.is_nullable(arg) {
                nullable = true;
            }
            args[dp] = arg;
            dp += 1;
            prev = arg;
        }
        args.truncate(dp);

        // TODO we should probably do sth similar in And
        if num_bytes > 1 {
            let mut byteset = vec![0u32; self.alphabet_words];
            args.retain(|&e| {
                let n = self.get(e);
                match n {
                    Expr::Byte(b) => {
                        byteset_set(&mut byteset, b as usize);
                        false
                    }
                    Expr::ByteSet(s) => {
                        byteset_union(&mut byteset, s);
                        false
                    }
                    _ => true,
                }
            });
            let node = self.mk_byte_set(&byteset);
            add_to_sorted(&mut args, node);
        }

        if num_lookahead > 1 {
            let mut lookahead = vec![];
            args.retain(|&e| {
                let n = self.get(e);
                match n {
                    Expr::Lookahead(_, inner, n) => {
                        lookahead.push((e, inner, n));
                        false
                    }
                    _ => true,
                }
            });
            lookahead.sort_by_key(|&(_, e, n)| (e.0, n));

            let mut prev = ExprRef::INVALID;
            for idx in 0..lookahead.len() {
                let (l, inner, _) = lookahead[idx];
                if inner == prev {
                    continue;
                }
                prev = inner;
                args.push(l);
            }

            args.sort_unstable();
        }

        if args.len() == 0 {
            ExprRef::NO_MATCH
        } else if args.len() == 1 {
            args[0]
        } else {
            let flags = ExprFlags::from_nullable(nullable);
            if self.optimize {
                self.or_optimized(flags, args)
            } else {
                self.mk(Expr::Or(flags, &args))
            }
        }
    }

    fn or_optimized(&mut self, flags: ExprFlags, args: Vec<ExprRef>) -> ExprRef {
        let mut concats: Vec<Vec<ExprRef>> = args
            .iter()
            .map(|e| self.flatten_tag(ExprTag::Concat, vec![*e]))
            .collect();
        concats.sort_unstable();
        let mut prev = ExprRef::INVALID;
        let mut has_double = false;
        for c in &concats {
            if c[0] == prev {
                has_double = true;
                break;
            }
            prev = c[0];
        }
        if !has_double {
            self.mk(Expr::Or(flags, &args))
        } else {
            self.optimize = false;
            let r = self.trie_rec(&concats, 0, 0);
            self.optimize = true;
            r
        }
    }

    // The idea is to optimize regexps like identifier1|identifier2|...|identifier50000
    // into a "trie" with shared prefixes;
    // for example: (foo|far|bar|baz) => (ba[rz]|f(oo|ar))
    fn trie_rec(&mut self, args: &[Vec<ExprRef>], offset: usize, depth: usize) -> ExprRef {
        // limit recursion depth
        if depth > 100 {
            let v = args
                .iter()
                .map(|v| self.mk_concat((&v[offset..]).to_vec()))
                .collect();
            return self.mk_or(v);
        }

        let mut end_offset = offset;
        let first = &args[0];
        let last = args.last().unwrap();
        let l = std::cmp::min(first.len(), last.len());
        while end_offset < l && first[end_offset] == last[end_offset] {
            end_offset += 1;
        }
        // println!(
        //     "depth: {} alen={} off={}-{}",
        //     depth,
        //     args.len(),
        //     offset,
        //     end_offset
        // );
        assert!(offset == 0 || end_offset > offset);
        let mut idx = 0;

        let mut alternatives = vec![];
        while idx < args.len() {
            let cur = args[idx].get(end_offset);
            let mut next = idx;
            while next < args.len() && args[next].get(end_offset) == cur {
                next += 1;
            }

            if cur.is_some() {
                alternatives.push(self.trie_rec(&args[idx..next], end_offset, depth + 1));
            } else {
                alternatives.push(ExprRef::EMPTY_STRING);
            }

            idx = next;
        }

        let mut children = first[offset..end_offset].to_vec();
        children.push(self.mk_or(alternatives));
        self.mk_concat(children)
    }

    pub fn mk_and(&mut self, mut args: Vec<ExprRef>) -> ExprRef {
        args = self.flatten_tag(ExprTag::And, args);
        self.pay(args.len());
        args.sort_unstable();
        let mut dp = 0;
        let mut prev = ExprRef::ANY_STRING;
        let mut had_empty = false;
        let mut nullable = true;
        for idx in 0..args.len() {
            let arg = args[idx];
            if arg == prev || arg == ExprRef::ANY_STRING {
                continue;
            }
            if arg == ExprRef::NO_MATCH {
                return ExprRef::NO_MATCH;
            }
            if arg == ExprRef::EMPTY_STRING {
                had_empty = true;
            }
            if nullable && !self.is_nullable(arg) {
                nullable = false;
            }
            args[dp] = arg;
            dp += 1;
            prev = arg;
        }
        args.truncate(dp);

        if args.len() == 0 {
            ExprRef::ANY_STRING
        } else if args.len() == 1 {
            args[0]
        } else if had_empty {
            if nullable {
                ExprRef::EMPTY_STRING
            } else {
                ExprRef::NO_MATCH
            }
        } else {
            let flags = ExprFlags::from_nullable(nullable);
            self.mk(Expr::And(flags, &args))
        }
    }

    pub fn mk_concat(&mut self, mut args: Vec<ExprRef>) -> ExprRef {
        args = self.flatten_tag(ExprTag::Concat, args);
        self.pay(args.len());
        args.retain(|&e| e != ExprRef::EMPTY_STRING);
        if args.len() == 0 {
            ExprRef::EMPTY_STRING
        } else if args.len() == 1 {
            args[0]
        } else if args.iter().any(|&e| e == ExprRef::NO_MATCH) {
            ExprRef::NO_MATCH
        } else {
            let flags = ExprFlags::from_nullable(args.iter().all(|&e| self.is_nullable(e)));
            self.mk(Expr::Concat(flags, &args))
        }
    }

    pub fn mk_byte_literal(&mut self, s: &[u8]) -> ExprRef {
        self.pay(s.len());
        let mut args = vec![];
        for b in s {
            args.push(self.mk_byte(*b));
        }
        self.mk_concat(args)
    }

    pub fn mk_literal(&mut self, s: &str) -> ExprRef {
        self.mk_byte_literal(s.as_bytes())
    }

    pub fn mk_not(&mut self, e: ExprRef) -> ExprRef {
        self.pay(1);
        if e == ExprRef::EMPTY_STRING {
            ExprRef::NON_EMPTY_STRING
        } else if e == ExprRef::NON_EMPTY_STRING {
            ExprRef::EMPTY_STRING
        } else if e == ExprRef::ANY_STRING {
            ExprRef::NO_MATCH
        } else if e == ExprRef::NO_MATCH {
            ExprRef::ANY_STRING
        } else {
            let n = self.get(e);
            match n {
                Expr::Not(_, e2) => return e2,
                _ => {}
            }
            let flags = ExprFlags::from_nullable(!n.nullable());
            self.mk(Expr::Not(flags, e))
        }
    }

    pub fn mk_lookahead(&mut self, mut e: ExprRef, offset: u32) -> ExprRef {
        self.pay(1);
        if e == ExprRef::NO_MATCH {
            return ExprRef::NO_MATCH;
        }

        let flags = if self.is_nullable(e) {
            e = ExprRef::EMPTY_STRING;
            ExprFlags::NULLABLE
        } else {
            ExprFlags::ZERO
        };
        self.mk(Expr::Lookahead(flags, e, offset))
    }
}

fn add_to_sorted(args: &mut Vec<ExprRef>, e: ExprRef) {
    let idx = args.binary_search(&e).unwrap_or_else(|x| x);
    assert!(idx == args.len() || args[idx] != e);
    args.insert(idx, e);
}
