use gvm_abi::{GuidanceVm, GuidanceVmHelper, expose, gvm_expose_all};

pub struct MyGvm {
    helper: GuidanceVmHelper,
}

impl GuidanceVm for MyGvm {
    fn gvm_create() -> Self {
        MyGvm {
            helper: GuidanceVmHelper::new(),
        }
    }

    fn gvm_process_prompt(&mut self) {}

    fn gvm_append_token(&mut self, token: u32) {
        let toks = &mut self.helper.tokens;
        toks.push(token);
        // finish generation at 10 tokens
        if toks.len() - self.helper.prompt_length >= 3 {
            self.helper.logit_biases[50256] = 100.0
        } else {
            self.helper.logit_biases[50256] = -100.0
        }
    }

    fn gvm_clone(&mut self) -> Self {
        MyGvm {
            helper: self.helper.clone(),
        }
    }
}

gvm_expose_all!(MyGvm);

fn main() {}
