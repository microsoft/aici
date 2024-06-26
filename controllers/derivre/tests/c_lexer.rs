use derivre::{RegexVec, StateID};

const C_SAMPLE: &str = r#"
#include "devs_internal.h"
#include "devs_objects.h"

// #define LOG_TAG "obj"
#include "devs_logging.h"

void devs_map_clear(devs_ctx_t *ctx, devs_map_t *map) {
    if (map->data) {
        devs_free(ctx, map->data);
        map->data = NULL;
        map->capacity = 0;
        map->length = 0;
    }
}

static inline uint16_t *short_keys(devs_short_map_t *map) {
    return (uint16_t *)(map->short_data + map->capacity);
}

static value_t *lookup_short(devs_ctx_t *ctx, devs_short_map_t *map, uint16_t key) {
    unsigned len = map->length;
    uint16_t *keys = short_keys(map);
    for (unsigned i = 0; i < len; i++) {
        if (keys[i] == key) {
            return &map->short_data[i];
        }
    }
    return NULL;
}
"#;

#[test]
fn c_lexer() {
    let regexps: Vec<_> = vec![
        r"/\*[^*]*\*+([^/*][^*]*\*+)*/", // block comment
        r"//.*",                         // line comment
        r"\n[ \t\v\f]*#(.*\\\n)*.*",     // pre-processor
        r"\n?[ \t\v\f]*",                // white-space
        r"[a-zA-Z_][0-9a-zA-Z_]*_t",     // type-name
        r"[a-zA-Z_][0-9a-zA-Z_]*",       // identifier
        r"0[xX][0-9a-fA-F]+[uUlL]*?",
        r"0[0-9]+[uUlL]*?",
        r"[0-9]+[uUlL]*?",
        r"[a-zA-Z_]?'(\\.|[^\\'])+'",
        r"[0-9]+[Ee][+-]?[0-9]+[flFL]?",
        r"[0-9]*\\.[0-9]+([Ee][+-]?[0-9]+)?[flFL]?",
        r"[0-9]+\\.[0-9]*([Ee][+-]?[0-9]+)?[flFL]?",
        r#"[a-zA-Z_]?"(\\.|[^\\"])*"#,
    ];
    let mut patterns: Vec<_> = vec![
        "!", "!=", "%", "%=", "&", "&&", "&=", "(", ")", "*", "*=", "+", "++", "+=", ",", "-",
        "--", "-=", "->", ".", "...", "/", ":", ";", "<", "<<", "<<=", "<=", "=", "==", ">", ">=",
        ">>", ">>=", "?", "[", "]", "^", "^=", "auto", "bool", "break", "case", "char", "const",
        "continue", "default", "do", "double", "else", "enum", "extern", "float", "for", "goto",
        "if", "inline", "int", "long", "register", "return", "short", "signed", "sizeof", "static",
        "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "{", "|",
        "|=", "||", "}", "~",
    ]
    .iter()
    .map(|s| regex_syntax::escape(s))
    .collect();
    patterns.extend(regexps.iter().map(|s| s.to_string()));
    let parser = regex_syntax::ParserBuilder::new()
        // .unicode(false)
        // .utf8(false)
        .build();
    let mut rx = RegexVec::new_with_parser(
        parser,
        &patterns.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    )
    .unwrap();
    let mut state = rx.initial_state_all();
    let mut start_idx = 0;
    let mut num_if = 0;
    for (idx, b) in C_SAMPLE.as_bytes().iter().enumerate() {
        let new_state = rx.transition(state, *b);
        if new_state == StateID::DEAD {
            let desc = rx.state_desc(state);
            if desc.is_accepting() {
                let lexeme = &patterns[desc.lowest_accepting.unwrap()];
                println!(
                    "matched: {:?} {:?}",
                    // desc,
                    lexeme,
                    &C_SAMPLE[start_idx..idx]
                );
                if patterns[desc.lowest_accepting.unwrap()] == "if" {
                    num_if += 1;
                }
                start_idx = idx;
                state = rx.initial_state_all();
                state = rx.transition(state, *b);
                if state == StateID::DEAD {
                    panic!("fail 1 ch");
                }
            } else {
                panic!("fail");
            }
        } else {
            state = new_state;
        }
    }
    println!("done; {:?}", rx);
    assert_eq!(num_if, 2);
}
