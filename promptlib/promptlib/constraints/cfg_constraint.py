from .constraint import Constraint

class ContextFreeGrammarConstraint(Constraint):

    class Rule:
        def __init__(self, symbol, productions):
            self.symbol = symbol
            # productions is an array of terminal and nonterminal symbols. 
            # if the array is empty, then the production is the empty string
            self.productions = productions 

        def is_empty_production(self):
            return len(self.productions) == 0
        
        def produces(self, symbol):
            return symbol in self.productions

        def __str__(self):
            return f"{self.symbol} -> {self.productions}"


    def __init__(self, grammar={}, curr_pos=None, start_symbol='S'):
        self.start_symbol = start_symbol
        self.grammar = grammar
        self.mtable = None # TODO change to _mtable, and note that finalize_grammar will create the mtable
        self._mtablekeys = None
        self.stack = []
        self.stack.append('$')
        self.stack.append(start_symbol)
    
    def _is_terminal(self, symbol):
        return symbol[0] == '@'


    # sometimes we want to run _first when we know the specific rule we want to start with.  This is a helper 
    # function for that, and also used inside the generic _first() function that just takes any symbol
    def _first_helper(self, symbol, rule, symbol_stack=[]):
        if rule.is_empty_production():
            return ['']

        ret = []
        num_empties = 0
        symbol_stack.append(symbol)
        for y in rule.productions:
            if y in symbol_stack:
                # don't get into an infinite recursion. recursing into a symbol we've already seen isn't useful
                # NOTE: this does mean that we cannot have recursive productions that are also empty productions 
                # TODO: when we don't recurse because of this we need to actually STOP not continue!
                #continue
                # TODO - double check do I have to do anything about ret[]? i don't want the wrong follows() to appear...
                break
            y_first = self._first(y, symbol_stack = symbol_stack)
            if '' not in y_first:
                ret.extend(y_first)
                break
            else:
                num_empties += 1
                y_first.remove('')
                ret.extend(y_first)
        popped_symbol = symbol_stack.pop()
        assert popped_symbol == symbol, "This shouldn't happen.  We should be popping the same symbol we pushed"

        return ret

    def _first(self, symbol, symbol_stack=[]):
        if self._is_terminal(symbol):
            return [symbol[1]]
        ret = []
        rules = self.grammar[symbol]
        # if any of the rules is an empty production, then add '' to ret
        for rule in rules:
            ret.extend(self._first_helper(symbol, rule, symbol_stack = symbol_stack))
            
        return set(ret)

    def _follow(self, symbol):
        # TODO - what do I do with terminals?  I'm not sure why the dragon book says this algo is only for non-terminals
        assert not self._is_terminal(symbol), "symbol must be a non-terminal"

        ret = []
        if( symbol == self.start_symbol ):
            ret.append('$')

        for rules in self.grammar.values():
            for rule in rules:
                if rule.produces(symbol):
                    for i, x in enumerate(rule.productions):
                        if x == symbol:
                            # get all terminals and nonterminals Y that follow symbol, and add first(Y) to ret (except for ''); if first(Y) contains '', then add follow(symbol) to ret
                            if( i + 1 < len(rule.productions) ):
                                add_follows = True
                                for y in rule.productions[i+1:]:
                                    y_first = self._first(y)
                                    if '' not in y_first:
                                        ret.extend(y_first)
                                        add_follows = False
                                        break
                                    else:
                                        y_first.remove('')
                                        ret.extend(y_first)
                                if add_follows:
                                    # TODO if symbol can be the last element in the production A->... b/c all the other productions on its right can be reduced to '' then add follows(A) to ret
                                    ret.extend(self._follow(rule.symbol))
                            else:
                                # if symbol is the last element in the production A->..., then add follows(A) to ret
                                ret.extend(self._follow(rule.symbol))                        

        return set(ret)

    def add_grammar_rule(self, rule):
        assert isinstance(rule, self.Rule), "rule must be a ContextFreeGrammarConstraint.Rule object"
        if( rule.symbol not in self.grammar ):
            self.grammar[rule.symbol] = []
        self.grammar[rule.symbol].append(rule)

    def finalize_grammar(self):
        self._build_mtable()

    def _build_mtable(self):
        self.mtable = {} # reset the mtable
        for nonterminal in self.grammar.keys():
            rules = self.grammar[nonterminal]
            for rule in rules:
                first = self._first_helper(nonterminal, rule)
                for terminal in first:
                    if (nonterminal, terminal) not in self.mtable.keys():
                        self.mtable[(nonterminal, terminal)] = []
                    self.mtable[(nonterminal, terminal)].append(rule)
                if '' in first:
                    follow = self._follow(nonterminal)
                    for terminal in follow:
                        if (nonterminal, terminal) not in self.mtable.keys():
                            self.mtable[(nonterminal, terminal)] = []
                        self.mtable[(nonterminal, terminal)] = rule

        self._mtablekeys = {}
        for nt,t in self.mtable.keys():
            if nt not in self._mtablekeys:
                self._mtablekeys[nt] = []
            self._mtablekeys[nt].append(t)

        for nt in self._mtablekeys.keys():
            self._mtablekeys[nt] = set(self._mtablekeys[nt])

    def valid_next_chars(self):
        assert self.mtable is not None, "grammar must be finalized before calling valid_next_chars"        
        assert self._mtablekeys is not None, "mtablekeys must be initialized before calling valid_next_chars"
            
        # see dragon book algo 4.3 p187
        X = self.stack[-1]
        if X == '$':
            return []
        elif self._is_terminal(X):
            # TODO if the symbol is a terminal, remember that we actually need to expand to its lexical production 
            return [X[1]]  ## TEMPORARILY, we are just returning the 2nd character because terminals are identified as beginning with '@'
        else: # X is a non-terminal
            assert X in self._mtablekeys, "This shouldn't happen. X (a non-terminal) must be in the mtablekeys.  This means this non-terminal is not defined in the grammar"
            return self._mtablekeys[X]

        pass

    def _set_next_chars_helper(self, c, new_stack):
        X = new_stack[-1]
        if X == '$':
            assert False, "This shouldn't happen.  We should never be trying to set the next chars when the stack is empty or $"
        elif self._is_terminal(X):
            if X[1] == c:
                new_stack.pop()
            else:
                assert False, f"next char '{c}' must match the terminal '{X[1]}' on the stack"
        else: # X is a non-terminal
            rule = self.mtable[(X, c)]
            new_stack.pop() # pop X from the stack
            # TODO - bug means that I can't have more than 1 rule per nonterminal right now. (hence the [0])
            #        what I need to do is change the mtable building so that the specific rule is saved for the mtable, and not all rules for the nonterminal
            for y in rule[0].productions[::-1]:
                new_stack.append(y) # push y onto the stack in reverse order
            # now that we've updated the stack, recurse until we get a terminal on top of the stack
            self._set_next_chars_helper(c, new_stack)

    def _set_next_char(self, c):
        # see dragon book algo 4.3 p187        
        new_stack = self.stack.copy()
        self._set_next_chars_helper(c, new_stack)

        ret = ContextFreeGrammarConstraint(grammar=self.grammar, start_symbol=self.start_symbol)
        ret.mtable = self.mtable
        ret._mtablekeys = self._mtablekeys
        ret.stack = new_stack
        return ret

    def set_next_chars(self, chars):
        ret = self
        for c in chars:
            ret = ret._set_next_char(c)
        return ret
