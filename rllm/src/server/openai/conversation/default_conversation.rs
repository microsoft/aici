use dyn_fmt::AsStrFormatExt;

use super::Conversation;

pub const ROLES: (&str, &str) = ("USER", "ASSISTANT");
pub const SYSTEM_TEMPLATE: &str = "{}";
pub const DEFAULT_SEP: &str = "\n";

/// Separator style for default conversation.
#[derive(Default)]
pub enum SeparatorStyle {
    #[default]
    AddColonSingle,
    AddColonTwo,
    AddColonSpaceSingle,
    NoColonSingle,
    NoColonTwo,
    AddNewLineSingle,
    Llama2,
    ChatGLM,
    ChatML,
    ChatIntern,
    Dolly,
    RWKV,
    Phoenix,
    Robin,
    FalconChat,
}

/// A struct for managing prompt templates and conversation history.
#[allow(dead_code)]
pub struct DefaultConversation {
    name: String,
    system_message: String,
    system_template: String,
    messages: Vec<Message>,
    offset: usize,
    sep_style: SeparatorStyle,
    stop_criteria: String,
    stop_token_ids: Vec<usize>,
    roles: (String, String),
    sep: String,
    sep2: Option<String>,
}

/// Default conversion separators
pub struct DefaultConversationSeparators {
    pub sep: String,
    pub sep2: Option<String>,
}

/// A message in a conversation
pub struct Message((String, Option<String>));

impl Message {
    pub fn new(message: (String, String)) -> Message {
        Message((message.0, Some(message.1)))
    }
}

impl DefaultConversation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        system_template: String,
        messages: Vec<Message>,
        offset: usize,
        sep_style: SeparatorStyle,
        stop_criteria: String,
        stop_token_ids: Vec<usize>,
        roles: (String, String),
        seps: DefaultConversationSeparators,
    ) -> Self {
        Self {
            name,
            system_message: "".to_string(),
            system_template,
            messages,
            offset,
            sep_style,
            stop_criteria,
            stop_token_ids,
            roles,
            sep: seps.sep,
            sep2: seps.sep2,
        }
    }
}

impl Conversation for DefaultConversation {
    /// Set the system message.
    fn set_system_message(&mut self, system_message: String) {
        self.system_message = system_message;
    }

    /// Append a new message.
    fn append_message(&mut self, role: String, message: String) {
        self.messages.push(Message((role, Some(message))));
    }

    /// Append a new `None` message.
    fn append_none_message(&mut self, role: String) {
        self.messages.push(Message((role, None)));
    }

    /// Set the last message to `None`.
    fn update_last_message(&mut self) {
        self.messages.last_mut().unwrap().0 .1 = None;
    }

    fn get_roles(&self) -> &(String, String) {
        &self.roles
    }

    /// Convert this conversation to a String prompt
    fn get_prompt(&mut self) -> String {
        let system_prompt = self.system_template.format(&[self.system_message.clone()]);
        match self.sep_style {
            SeparatorStyle::AddColonSingle => {
                let mut accum = system_prompt + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeparatorStyle::AddColonTwo => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = system_prompt + &self.sep;
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", seps[i % 2]);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeparatorStyle::AddColonSpaceSingle => {
                let mut accum = system_prompt + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}: "); //must end with space
                    }
                }
                accum
            }

            SeparatorStyle::AddNewLineSingle => {
                let mut accum = if system_prompt.is_empty() {
                    "".to_string()
                } else {
                    system_prompt.clone() + &self.sep
                };
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}\n{message}{}", self.sep);
                    } else {
                        accum += &format!("{role}\n");
                    }
                }
                accum
            }

            SeparatorStyle::NoColonSingle => {
                let mut accum = system_prompt.clone();
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}{message}{}", self.sep);
                    } else {
                        accum += role;
                    }
                }
                accum
            }

            SeparatorStyle::NoColonTwo => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = system_prompt.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}{message}{}", seps[i % 2]);
                    } else {
                        accum += role;
                    }
                }
                accum
            }

            SeparatorStyle::RWKV => {
                let mut accum = system_prompt.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!(
                            "{role}: {}\n\n",
                            message.replace("\r\n", "\n").replace("\n\n", "\n")
                        );
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeparatorStyle::Llama2 => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = if !system_prompt.is_empty() {
                    system_prompt.clone()
                } else {
                    "[INST] ".to_string()
                };
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((_role, message)) = message;

                    let tag = &[self.roles.0.clone(), self.roles.1.clone()][i % 2];

                    if let Some(message) = message {
                        if i == 0 {
                            accum += &format!("{message} ");
                        } else {
                            accum += &format!("{tag} {message}{}", seps[i % 2]);
                        }
                    } else {
                        accum += tag;
                    }
                }
                accum
            }

            SeparatorStyle::ChatGLM => {
                let round_add_n = if self.name == "chatglm2" { 1 } else { 0 };

                let mut accum = if !system_prompt.is_empty() {
                    system_prompt.clone()
                } else {
                    "".to_string()
                };

                for (i, message) in self.messages.iter().enumerate() {
                    if i % 2 == 0 {
                        accum += &format!("[Round {}]{}", i / 2 + round_add_n, self.sep);
                    }
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}: ");
                    }
                }
                accum
            }

            SeparatorStyle::ChatML => {
                let mut accum = if !system_prompt.is_empty() {
                    format!("{}{}\n", system_prompt, self.sep)
                } else {
                    "".to_string()
                };
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}\n{message}{}\n", self.sep);
                    } else {
                        accum += &format!("{role}\n");
                    }
                }
                accum
            }

            SeparatorStyle::ChatIntern => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = system_prompt.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    if i % 2 == 0 {
                        accum += "<s>";
                    }

                    let Message((role, message)) = message;

                    if let Some(message) = message {
                        accum += &format!("{role}:{message}{}\n", seps[i % 2]);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }

            SeparatorStyle::Dolly => {
                let seps = [&self.sep, &self.sep2.clone().unwrap_or("".to_string())];
                let mut accum = system_prompt.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    let Message((role, message)) = message;

                    if let Some(message) = message {
                        accum += &format!("{role}:\n{message}{}", seps[i % 2]);
                        if i % 2 == 1 {
                            accum += "\n\n";
                        }
                    } else {
                        accum += &format!("{role}:\n");
                    }
                }
                accum
            }

            SeparatorStyle::Phoenix => {
                let mut accum = system_prompt.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: <s>{message}</s>");
                    } else {
                        accum += &format!("{role}: <s>");
                    }
                }
                accum
            }

            SeparatorStyle::Robin => {
                let mut accum = system_prompt.clone() + &self.sep;
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}:\n{message}{}", self.sep);
                    } else {
                        accum += &format!("{role}:\n");
                    }
                }
                accum
            }

            SeparatorStyle::FalconChat => {
                let mut accum = "".to_string();
                if !system_prompt.is_empty() {
                    accum += &format!("{}{}", system_prompt, self.sep)
                }
                for message in &self.messages {
                    let Message((role, message)) = message;
                    if let Some(message) = message {
                        accum += &format!("{role}: {message}{}", self.sep);
                    } else {
                        accum += &format!("{role}:");
                    }
                }
                accum
            }
        }
    }
}
