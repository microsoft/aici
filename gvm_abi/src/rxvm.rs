use crate::rx::{StateOffset, TokRx};
use crate::{println, GuidanceVm, GuidanceVmHelper};

pub struct RxGvm {
    pub helper: GuidanceVmHelper,
    pub compiled: TokRx,
    pub state: StateOffset,
}

impl RxGvm {
    pub fn from_token_compiled(compiled: TokRx) -> Self {
        RxGvm {
            helper: GuidanceVmHelper::new(),
            compiled,
            state: StateOffset::START,
        }
    }
}

impl GuidanceVm for RxGvm {
    fn gvm_process_prompt(&mut self) {
        println!("prompt, {} tokens", self.helper.prompt_length);
        // the regex doesn't care about the prompt
        self.state = StateOffset::START;
        self.compiled
            .compute_logit_bias(self.state, &mut self.helper.logit_biases);
    }

    fn gvm_append_token(&mut self, token: u32) {
        // println!("xapp {:?} {} {}", self as *const _, token, self.state.off);
        self.state = self.compiled.advance(self.state, token as u16);

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        // compute biases
        self.compiled
            .compute_logit_bias(self.state, &mut self.helper.logit_biases);
    }

    // implement by hand for now - we may need some special processing here
    fn gvm_clone(&mut self) -> Self {
        let r = RxGvm {
            helper: self.helper.clone(),
            compiled: self.compiled.clone(),
            state: self.state.clone(),
        };
        println!("{} -> {}", self.state.off, r.state.off);
        r
    }
}
