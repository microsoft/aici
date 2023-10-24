# Implementation notes

## Token trie

The round nodes represent tokens, the square nodes do not have a corresponding token.

The number (`num_parents`) specifies how many parents do you need to pop to get to the parent of the node which comes after our children in DFS order.

We also keep the `token_id` and a `subtree_size` (which includes the node itself) in each node.
A bogus `token_id` is used for nodes that do not have a corresponding token.

```mermaid
graph TD
  root[ε, 0] -- a --> a((a, 1))
  root -- b --> b((b, 1))
  root -- c --> c((c, 1))
  a -- x --> ax((ax, 1))
  a -- y --> ay[ay, 1]
  a -- z --> az((az, 2))
  az -- a --> azq((aza, 3))
  ay -- a --> ayq((aya, 1))
  ay -- b --> ayw((ayb, 2))
```

Traversal algorithm - computing the set of tokens allowed by a stack-based recognizer.
The set is stored in `logits` array - entries with `0.0` are allowed.

```rust
let mut logits = vec![-100.0; VOCAB_SIZE + 1];
```

A simple version of traversal algorithm:

```rust
fn traverse(n) {
    // mark token as allowed; nodes without token use `token_id == VOCAB_SIZE`
    logits[n.token_id] = 0.0;
    for c in n.children {
        // for every child that starts with an allowed byte
        if byte_allowed(c.byte) {
            push_byte(c.byte);
            // traverse it
            traverse(c);
            pop_bytes(1);
        }
    }
}
```

Now, assume the tree is laid out in memory in DFS order:

```rust
fn traverse(mut p) {
    let endp = p + nodes[p].subtree_size;
    p += 1; // move to first child
    while p < endp {
        let n = nodes[p];
        if byte_allowed(n.byte) {
            push_byte(n.byte);
            logits[n.token_id] = 0.0;
            // p is moved by n.subtree_size
            p = traverse(p);
            pop_bytes(1);
        } else {
            p += n.subtree_size;
        }
    }
}
```

Now, we get rid of the recursion:

```rust
let mut p = 0;
while p < nodes.len() {
    let n = nodes[p];
    if byte_allowed(n.byte) {
        push_byte(n.byte);
        logits[n.token_id] = 0.0;
        // if the node is a leaf, we need to pop all the parents
        pop_bytes(if n.subtree_size == 1 { n.num_parents } else { 0 });
        // move to first child, or sibling if no children
        p += 1;
    } else {
        // skip the children, and go to the sibling node
        p += n.subtree_size;
        // regardless if the node is a leaf, we need to pop all the parents
        pop_bytes(n.num_parents - 1);
    }
}
```

Note that the only branch that gets mis-predicted here is the `if byte_allowed(n.byte)`.
The `if` in argument to `pop_bytes` is compiled to bit operations, so it is branchless.
