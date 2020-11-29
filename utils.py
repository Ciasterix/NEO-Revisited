# import tensorflow as tf
import io
import tokenize


class TreeTokenizer:

    def __init__(self, pset):
        self.pset = pset
        self.terminals = {t.name: t for t in pset.terminals[pset.ret]}
        self.primitives = {p.name: p for p in pset.primitives[pset.ret]}
        self.vocabulary = [*self.terminals.keys()] + [*self.primitives.keys()]
        self.tokens2id = {t: i for i, t in enumerate(self.vocabulary)}
        self.id2tokens = {i: t for t, i in self.tokens2id.items()}
        self.trans = str.maketrans({'(': ' ', ')': ' ', ',': ' '})

    def _tokenize(self, string_tree):
        tokens_string = string_tree.translate(self.trans)
        code = io.BytesIO(tokens_string.encode())
        tokens = list(tokenize.tokenize(code.readline))
        tokens = [t.string for t in tokens if t.string in self.vocabulary]
        return tokens

    def tokenize_tree(self, string_tree):
        tokens = self._tokenize(string_tree)
        ids = [self.tokens2id[t] for t in tokens]
        return ids

    def reproduce_expression(self, tokens):
        tokens_names = [self.id2tokens[t] for t in tokens]
        print('tokens_names:', tokens_names)
        expr = []
        for tn in tokens_names:
            if tn in self.terminals:
                expr.append(self.terminals[tn])
            if tn in self.primitives:
                expr.append(self.primitives[tn])
        return expr

    def validate_expression(self, tokens, expr):
        return len(tokens) == len(self._tokenize(expr))


if __name__ == "__main__":
    import benchmarks
    from deap import gp

    pset = benchmarks.standard_boolean_pset(num_in=6)
    tokenizer = TreeTokenizer(pset)
    print("Setup tokenizer:")
    print("tokens2id:")
    print(tokenizer.tokens2id)
    print("id2tokens:")
    print(tokenizer.id2tokens)

    s2 = 'nand(nand(IN3, IN1), or_(nand(and_(IN2, IN0), '\
         + 'or_(IN2, IN3)), nand(IN4, nand(IN5, 0))))'

    print("\nTokenizing string")
    print(s2)
    print("Tokens names:")
    print(tokenizer._tokenize(s2))
    print("Tokens ids:")
    tokens = tokenizer.tokenize_tree(s2)
    print(tokens)

    print("\nnReproducing Tree")
    expr = tokenizer.reproduce_expression(tokens)
    print("Expression")
    print(expr)
    print("Tree:")
    tree = gp.PrimitiveTree(expr)
    print(str(tree))

    print("\nValidate expression:")
    print(tokenizer.validate_expression(tokens, str(tree)))

    print("\n--------------- INVALID EXAMPLE -----------")
    invalid_tokens = tokens[1:]

    print("\nnReproducing Invalid Tree")
    invalid_expr = tokenizer.reproduce_expression(invalid_tokens)
    print("Expression")
    print(invalid_expr)
    print("Tree:")
    invalid_tree = gp.PrimitiveTree(invalid_expr)
    print(str(invalid_tree))
    print("Validate invalid expression:")
    print(tokenizer.validate_expression(invalid_tokens, str(invalid_tree)))
