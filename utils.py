# import tensorflow as tf
import tokenize, io


class TreeTokenizer:

    def __init__(self, pset):
        self.pset = pset
        terminals = [p.format() for p in self.pset.terminals[self.pset.ret]]
        primitives = [p.name for p in self.pset.primitives[self.pset.ret]]
        self.vocabulary = terminals + primitives
        self.tokens2id = {t: i for i, t in enumerate(self.vocabulary)}
        self.id2tokens = {i: t for t, i in self.tokens2id.items()}
        self.trans = str.maketrans({'(': ' ', ')': ' ', ',': ' '})

    def tokenize(self, string_tree):
        tokens_string = string_tree.translate(self.trans)
        tokens = list(
            tokenize.tokenize(io.BytesIO(tokens_string.encode()).readline))
        tokens = [t.string for t in tokens if t.string in self.vocabulary]
        return tokens

    def tree_to_ids(self, string_tree):
        tokens = self.tokenize(string_tree)
        ids = [self.tokens2id[t] for t in tokens]
        return ids

    def validate_program(self, program):
        try:
            eval(program)
        except:
            return False
        return True


if __name__ == "__main__":
    import benchmarks
    pset = benchmarks.standard_boolean_pset(num_in=6)
    tokenizer = TreeTokenizer(pset)
    print("tokens2id:", tokenizer.tokens2id)
    print("id2tokens:", tokenizer.id2tokens)

    s2 = 'nand(nand(IN3, IN1), or_(nand(and_(IN2, IN0),\
     or_(IN2, IN3)), nand(IN4, nand(IN5, 0))))'

    print("Tokenizing second string")
    print(s2)
    print(tokenizer.tokenize(s2))
    print(tokenizer.tree_to_ids(s2))
