# import tensorflow as tf
import io
import tokenize


class TreeTokenizer:

    def __init__(self, pset, max_size):
        self.pset = pset
        self.max_size = max_size

        self.terminals = {t.name: t for t in pset.terminals[pset.ret]}
        self.primitives = {p.name: p for p in pset.primitives[pset.ret]}

        self.vocabulary = ['<pad>', '<start>', '<end>'] +\
                          [*self.terminals.keys()] + [*self.primitives.keys()]
        self.tokens2id = {t: i for i, t in enumerate(self.vocabulary)}
        self.id2tokens = {i: t for t, i in self.tokens2id.items()}

        # TODO check if this trans can be deleted
        self.trans = str.maketrans({'(': ' ', ')': ' ', ',': ' '})

    def __pad(self, _list):
        _list.extend([0] * (self.max_size - len(_list)))

    def _tokenize(self, string_tree):
        tokens_string = string_tree.translate(self.trans)
        code = io.BytesIO(tokens_string.encode())
        tokens = list(tokenize.tokenize(code.readline))
        tokens = [t.string for t in tokens if t.string in self.vocabulary]
        return tokens

    def tokenize_tree(self, string_tree):
        tokens = self._tokenize(string_tree)
        if len(tokens) > self.max_size - 2:
            print(string_tree)
            print(len(tokens))
            raise ValueError("Tree size bigger than tokenizer's max_size")
        tokens = ['<start>'] + tokens + ['<end>']
        ids_only = [self.tokens2id[t] for t in tokens]
        self.__pad(ids_only)
        return ids_only

    def reproduce_expression(self, tokens):
        tokens_names = [self.id2tokens[t] for t in tokens]
        expr = []
        if not tokens_names[0] == '<start>':
            raise ValueError(
                "First token is not '<start>' but " + tokens_names[0] )

        for tn in tokens_names[1:]:
            if tn == '<end>':
                break
            elif tn in self.terminals:
                expr.append(self.terminals[tn])
            elif tn in self.primitives:
                expr.append(self.primitives[tn])
            else:
                raise ValueError("Wrong token:" + str(tn))
        return expr

    def validate_expression(self, tokens, expr):
        return len(tokens) == len(self._tokenize(expr))


if __name__ == "__main__":
    import benchmarks
    from deap import gp

    pset = benchmarks.standard_boolean_pset(num_in=6)
    tokenizer = TreeTokenizer(pset, 100)
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
