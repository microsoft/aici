
class Constraint:

    def __init__(self):
        pass

    def set_context(self, prefix, lm):
        pass

    def valid_next_chars(self):  ## can return nothing if generation is complete
        pass

    def set_next_chars(self, c): ## returns a new immutable Constraint object with an updated internal state, if state is needed
        return self
