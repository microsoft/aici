use serde::{Deserialize, Serialize};

/// This represents a collection of grammars, with a designated
/// "start" grammar at first position.
/// Grammars can refer to each other via GrammarRef nodes.
#[derive(Serialize, Deserialize)]
pub struct TopLevelGrammar {
    pub grammars: Vec<GrammarWithLexer>,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct GrammarWithLexer {
    /// The start symbol is at nodes[0]
    pub nodes: Vec<Node>,

    /// When enabled, the grammar can use `Lexeme` but not `Gen`.
    /// When disabled, the grammar can use `Gen` but not `Lexeme`.
    /// `String` is allowed in either case as a shorthand for either `Lexeme` or `Gen`.
    #[serde(default)]
    pub greedy_lexer: bool,

    /// Only applies to greedy_lexer grammars.
    /// This adds a new lexeme that will be ignored when parsing.
    pub greedy_skip_rx: Option<RegexSpec>,
}

#[derive(Serialize, Deserialize)]
pub enum Node {
    // Terminals:
    /// Force generation of the specific string.
    String {
        literal: String,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Generate according to regex.
    Gen {
        #[serde(flatten)]
        data: GenOptions,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Lexeme in a greedy grammar.
    Lexeme {
        /// The regular expression that will greedily match the input.
        rx: RegexSpec,

        /// When false, when these lexeme is recognized, all other lexemes are excluded.
        /// This is normal behavior for keywords in programming languages.
        /// Set to true for eg. a JSON schema with both `/"type"/` and `/"[^"]*"/` as lexemes.
        #[serde(default)]
        allow_others: bool,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Generate according to specified grammar.
    GenGrammar {
        #[serde(flatten)]
        data: GenGrammarOptions,

        #[serde(flatten)]
        props: NodeProps,
    },

    // Non-terminals:
    /// Generate one of the options.
    Select {
        among: Vec<NodeId>,

        #[serde(flatten)]
        props: NodeProps,
    },
    /// Generate all of the nodes in sequence.
    Join {
        sequence: Vec<NodeId>,

        #[serde(flatten)]
        props: NodeProps,
    },
}

/// Optional fields allowed on any Node
#[derive(Serialize, Deserialize)]
pub struct NodeProps {
    pub max_tokens: Option<usize>,
    pub name: Option<String>,
    pub capture_name: Option<String>,
}

pub type RegexSpec = String;

#[derive(Serialize, Deserialize)]
pub struct GenOptions {
    /// Regular expression matching the body of generation.
    pub body_rx: RegexSpec,

    /// The whole generation must match `body_rx + stop_rx`.
    /// Whatever matched `stop_rx` is discarded.
    /// If `stop_rx` is empty, it's assumed to be EOS.
    pub stop_rx: RegexSpec,

    /// Override sampling temperature.
    pub temperature: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenGrammarOptions {
    pub grammar: GrammarId,

    /// Add a lexeme that causes the generation to stop.
    pub stop_rx: Option<RegexSpec>,

    /// When set to true, the greedy_skip_rx of the grammar is ignored
    /// at the beginning of generation.
    #[serde(default)]
    pub no_initial_skip: bool,

    /// Override sampling temperature.
    pub temperature: Option<f32>,
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Serialize, Deserialize, Hash, PartialEq, Eq, Clone, Copy, Debug)]
        #[serde(transparent)]
        pub struct $name(pub usize);
    };
}

id_type!(GrammarId);
id_type!(NodeId);

impl Node {
    pub fn node_props(&self) -> &NodeProps {
        match self {
            Node::String { props, .. } => props,
            Node::Gen { props, .. } => props,
            Node::Lexeme { props, .. } => props,
            Node::GenGrammar { props, .. } => props,
            Node::Select { props, .. } => props,
            Node::Join { props, .. } => props,
        }
    }
}
