use aici_abi::{aici_expose_all, AiciVm, AiciVmHelper};

pub struct MyAici {
    helper: AiciVmHelper,
}

fn create() -> MyAici {
    MyAici {
        helper: AiciVmHelper::new(),
    }
}

impl AiciVm for MyAici {
    fn aici_process_prompt(&mut self) {}

    fn aici_append_token(&mut self, token: u32) {
        let toks = &mut self.helper.tokens;
        toks.push(token);
        // finish generation at 10 tokens
        if toks.len() - self.helper.prompt_length >= 3 {
            self.helper.logit_biases[50256] = 100.0
        } else {
            self.helper.logit_biases[50256] = -100.0
        }
    }

    fn aici_clone(&mut self) -> Self {
        MyAici {
            helper: self.helper.clone(),
        }
    }

    fn get_helper(&mut self) -> &mut AiciVmHelper {
        &mut self.helper
    }
}

aici_expose_all!(MyAici, create());

fn main() {}
