use crate::rx::{StateOffset, TokRx};
use crate::{wprintln, AiciVm, AiciVmHelper};

pub struct RxAici {
    pub helper: AiciVmHelper,
    pub compiled: TokRx,
    pub state: StateOffset,
}

impl RxAici {
    pub fn from_token_compiled(compiled: TokRx) -> Self {
        RxAici {
            helper: AiciVmHelper::new(),
            compiled,
            state: StateOffset::START,
        }
    }
}

impl AiciVm for RxAici {
    fn aici_process_prompt(&mut self) {
        wprintln!("prompt, {} tokens", self.helper.prompt_length);
        // the regex doesn't care about the prompt
        self.state = StateOffset::START;
        self.compiled
            .compute_logit_bias(self.state, &mut self.helper.logit_biases);
    }

    fn aici_append_token(&mut self, token: u32) {
        // wprintln!("xapp {:?} {} {}", self as *const _, token, self.state.off);
        self.state = self.compiled.advance(self.state, token);

        // save the token, just in case
        let toks = &mut self.helper.tokens;
        toks.push(token);

        // compute biases
        self.compiled
            .compute_logit_bias(self.state, &mut self.helper.logit_biases);
    }

    // implement by hand for now - we may need some special processing here
    fn aici_clone(&mut self) -> Self {
        let r = RxAici {
            helper: self.helper.clone(),
            compiled: self.compiled.clone(),
            state: self.state.clone(),
        };
        wprintln!("{} -> {}", self.state.off, r.state.off);
        r
    }

    fn get_helper(&mut self) -> &mut AiciVmHelper {
        &mut self.helper
    }
}
