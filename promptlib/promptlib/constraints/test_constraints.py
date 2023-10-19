from .constraint import Constraint

class DummyCharacterConstraint(Constraint):

    def __init__(self, positiveConstraint = ['a','b','c']):
        self.positiveConstraint = positiveConstraint
        #self.negativeConstraint = negativeConstraint

# TODO - playing around with what weighted constraints might look like.  Will come back to this after logprobs is added into the constrain() func
#    def __init__(self, positiveConstraint = ['a','b','c'], negativeConstraint = []):
#        self.constraint = { c: 1.0 for c in positiveConstraint }
#        self.constraint.update({ c: 0.0 for c in negativeConstraint })

    def valid_next_chars(self):
        return self.positiveConstraint



class OddEvenDigitConstraint(Constraint):

    def __init__(self, prevDigitWasEven=None):
        self.prevDigitWasEven = prevDigitWasEven
        self.evens = ['0','2','4','6','8']
        self.odds = ['1','3','5','7','9']
        self.all = self.evens + self.odds

    def valid_next_chars(self):
        if( self.prevDigitWasEven is None ):
            return self.all
        elif self.prevDigitWasEven:
            return self.odds
        else:
            return self.evens
        
    def set_next_chars(self, c):
        if( c[-1] in self.odds ):
            return OddEvenDigitConstraint(prevDigitWasEven = False)
        else:
            return OddEvenDigitConstraint(prevDigitWasEven = True)