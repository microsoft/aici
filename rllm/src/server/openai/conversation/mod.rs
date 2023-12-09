pub mod default_conversation;

/// A trait for using conversation managers with a `ModulePipeline`.
pub trait Conversation {
    fn set_system_message(&mut self, system_message: String);

    fn append_message(&mut self, role: String, message: String);

    fn append_none_message(&mut self, role: String);

    fn update_last_message(&mut self);

    fn get_roles(&self) -> &(String, String);

    fn get_prompt(&mut self) -> String;
}
