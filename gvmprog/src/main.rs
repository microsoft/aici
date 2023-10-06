use gvm_abi::{gvm_expose_all, GuidanceVm, GuidanceVmHelper};

pub struct MyGvm {
    helper: GuidanceVmHelper,
}

fn create() -> MyGvm {
    MyGvm {
        helper: GuidanceVmHelper::new(),
    }
}

impl GuidanceVm for MyGvm {
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

    fn get_helper(&mut self) -> &mut GuidanceVmHelper {
        &mut self.helper
    }
}

gvm_expose_all!(MyGvm, create());

fn main() {}
