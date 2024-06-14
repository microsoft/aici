use serde::{Deserialize, Serialize};

/// This represents a collection of grammars, with a designated
/// "start" grammar at first position.
/// Grammars can refer to each other via GrammarRef nodes.
#[derive(Serialize, Deserialize)]
pub struct TopLevelGrammar {
    pub grammars: Vec<GrammarWithLexer>,
    pub max_tokens: Option<usize>,
}

pub const DEFAULT_CONTEXTUAL: bool = true;

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

    /// The default value for 'contextual' in Lexeme nodes.
    pub contextual: Option<bool>,

    /// When set, the regexps can be referenced by their id (position in this list).
    #[serde(default)]
    pub rx_nodes: Vec<RegexNode>,
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

        /// If false, all other lexemes are excluded when this lexeme is recognized.
        /// This is normal behavior for keywords in programming languages.
        /// Set to true for eg. a JSON schema with both `/"type"/` and `/"[^"]*"/` as lexemes,
        /// or for "get"/"set" contextual keywords in C#.
        /// Default value set in GrammarWithLexer.
        contextual: Option<bool>,

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

    /// Override sampling temperature.
    pub temperature: Option<f32>,

    #[serde(skip)]
    pub max_tokens_grm: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum RegexNode {
    /// Intersection of the regexes
    And(Vec<RegexId>),
    /// Union of the regexes
    Or(Vec<RegexId>),
    /// Concatenation of the regexes
    Concat(Vec<RegexId>),
    /// Matches the regex; should be at the end of the main regex.
    /// The length of the lookahead can be recovered from the engine.
    LookAhead(RegexId),
    /// Matches everything the regex doesn't match.
    /// Can lead to invalid utf8.
    Not(RegexId),
    /// Repeat the regex at least min times, at most max times
    Repeat(RegexId, u32, Option<u32>),
    /// Matches the empty string. Same as Concat([]).
    EmptyString,
    /// Matches nothing. Same as Or([]).
    NoMatch,
    /// Compile the regex using the regex_syntax crate
    Regex(String),
    /// Matches this string only
    Literal(String),
    /// Matches this string of bytes only. Can lead to invalid utf8.
    ByteLiteral(Vec<u8>),
    /// Matches this byte only. If byte is not in 0..127, it may lead to invalid utf8
    Byte(u8),
    /// Matches any byte in the set, expressed as bitset.
    /// Can lead to invalid utf8 if the set is not a subset of 0..127
    ByteSet(Vec<u32>),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum RegexSpec {
    RegexId(RegexId),
    Regex(String),
}

impl RegexSpec {
    pub fn is_missing(&self) -> bool {
        match self {
            RegexSpec::RegexId(_) => false,
            RegexSpec::Regex(s) => s.is_empty(),
        }
    }
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
id_type!(RegexId);

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

impl Default for GenGrammarOptions {
    fn default() -> Self {
        GenGrammarOptions {
            grammar: GrammarId(0),
            temperature: None,
            max_tokens_grm: usize::MAX,
        }
    }
}
