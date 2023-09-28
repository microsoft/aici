use serde_json::Value;

fn json_to_regex_inner(json: &Value) -> String {
    let strrx = r#""(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*""#;
    match json {
        Value::Bool(_) => r#"(true|false)"#.into(),
        Value::Number(_) => r#"\d+"#.into(),
        Value::String(s) => {
            if s == "" {
                strrx.into()
            } else {
                format!("\"({})\"", s)
            }
        }
        Value::Array(_) => r#"\[.*\]"#.into(),
        Value::Object(obj) => {
            String::from(r#"\{\n?"#)
                + &obj
                    .iter()
                    .map(|(k, v)| format!("\"{0}\": {1}", k, json_to_regex_inner(v)))
                    .collect::<Vec<_>>()
                    .join(",\\n")
                + r#"\n?\}"#
        }
        Value::Null => r#"null"#.into(),
    }
}

pub fn json_to_regex(json: &Value) -> String {
    format!("{}", json_to_regex_inner(json))
}
