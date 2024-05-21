use serde::{Deserialize, Serialize};

/// This represents a collection of grammars, with a designated
/// "start" grammar at first position.
/// Grammars can refer to each other via GrammarRef symbols.
#[derive(Serialize, Deserialize)]
pub struct TopLevelGrammar {
    pub grammars: Vec<Grammar>,
}

#[derive(Serialize, Deserialize)]
pub struct Grammar {
    /// The start symbol is at symbols[0]
    pub symbols: Vec<Symbol>,

    /// When enabled, the grammar can use `Lexeme` but not `Gen`.
    /// When disabled, the grammar can use `Gen` but not `Lexeme`.
    /// `String` is allowed in either case as a shorthand for either `Lexeme` or `Gen`.
    #[serde(default)]
    pub greedy_lexer: bool,
}

#[derive(Serialize, Deserialize)]
pub enum Symbol {
    // Terminals:
    /// Force generation of the specific string.
    String {
        literal: String,

        #[serde(flatten)]
        props: SymbolProps,
    },
    /// Generate according to regex.
    Gen {
        #[serde(flatten)]
        data: GenOptions,

        #[serde(flatten)]
        props: SymbolProps,
    },
    /// Lexeme in a greedy grammar.
    Lexeme {
        /// The regular expression that will greedily match the input.
        rx: String,

        /// When false, when these lexeme is recognized, all other lexemes are excluded.
        /// This is normal behavior for keywords in programming languages.
        /// Set to true for eg. a JSON schema with both `/"type"/` and `/"[^"]*"/` as lexemes.
        #[serde(default)]
        allow_others: bool,

        #[serde(flatten)]
        props: SymbolProps,
    },
    /// Generate according to specified grammar.
    GrammarRef {
        grammar_id: GrammarId,

        #[serde(flatten)]
        props: SymbolProps,
    },

    // Non-terminals:
    /// Generate one of the options.
    Select {
        among: Vec<SymbolId>,

        #[serde(flatten)]
        props: SymbolProps,
    },
    /// Generate all of the symbols in sequence.
    Join {
        sequence: Vec<SymbolId>,

        #[serde(flatten)]
        props: SymbolProps,
    },
}

/// Optional fields allowed on any Symbol
#[derive(Serialize, Deserialize)]
pub struct SymbolProps {
    pub max_tokens: Option<usize>,
    pub name: Option<String>,
    pub capture_name: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct GenOptions {
    /// Regular expression matching the body of generation.
    pub body_rx: String,

    /// The whole generation must match `body_rx + stop_rx`.
    /// Whatever matched `stop_rx` is discarded.
    pub stop_rx: String,

    /// Override sampling temperature.
    pub temperature: Option<f32>,
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub usize);
    };
}

id_type!(GrammarId);
id_type!(SymbolId);
