use anyhow::Result;

struct StackNode<'a, T, S> {
    ast: &'a T,
    trg: usize,
    args: Vec<S>,
}

pub fn map_ast<'a, T, S>(
    ast: &'a T,
    get_args: impl Fn(&T) -> &[T],
    mut map_node: impl FnMut(&T, Vec<S>) -> Result<S>,
) -> Result<S> {
    let mut stack = vec![StackNode {
        ast,
        trg: 0,
        args: Vec::new(),
    }];

    while let Some(entry) = stack.pop() {
        let args = get_args(entry.ast);
        if args.len() > 0 && entry.args.len() == 0 {
            // children not yet processed
            let trg = stack.len();
            // re-push current
            stack.push(entry);
            // and children
            for ast in args.iter().rev() {
                stack.push(StackNode {
                    ast,
                    trg,
                    args: Vec::new(),
                });
            }
        } else {
            assert!(entry.args.len() == args.len());
            let r = map_node(entry.ast, entry.args)?;
            if stack.len() == 0 {
                return Ok(r);
            }
            stack[entry.trg].args.push(r);
        }
    }

    unreachable!()
}
