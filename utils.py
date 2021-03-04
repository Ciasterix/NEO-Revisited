# import tensorflow as tf
import io
import random
import tokenize


class TreeTokenizer:

    def __init__(self, pset, max_size):
        self.pset = pset
        self.max_size = max_size

        self.terminals = {t.name: t for t in pset.terminals[pset.ret]}
        self.primitives = {p.name: p for p in pset.primitives[pset.ret]}

        self.vocabulary = ['<pad>', '<start>', '<end>'] + \
                          [*self.terminals.keys()] + [*self.primitives.keys()]
        self.tokens2id = {t: i for i, t in enumerate(self.vocabulary)}
        print(self.tokens2id)
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
                "First token is not '<start>' but " + tokens_names[0])

        for tn in tokens_names[1:]:
            if tn == '<end>' or tn == '<pad>':
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

    s2 = 'nand(nand(IN3, IN1), or_(nand(and_(IN2, IN0), ' \
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

"""
Queue ADT: ArrayQueue

This class implements the following methods:
    Q.enqueue(e)
    Q.dequeue()
    Q.first()
    Q.is_empty()
    len(Q)

This class utilizes an array-based data structure.

The Queue class needs to keep track of where the
beginning and end of the structure is.
This will dynamically expand under the hood
as needed. This is an O(1) data structure,
when amortized.

It needs to keep track of three things:
- pointer to first item in queue
- number of elements in queue (use in place of last pointer)
- pointer to data structure itself

"""


class Empty(Exception):
    pass


class ArrayQueue:

    def __init__(self):
        """Keep track of 3 things"""
        INIT_CAP = 10
        self._data = [None] * INIT_CAP
        self._front = 0
        self._n = 0

    def __len__(self):
        """Return number of elements in queue"""
        return self._n

    def __str__(self):
        return str(self._data)
        # ix_sequence = [(self._front+i)%(len(self._data)) for i in range(self._n)]
        # contents = ",".join([str(self._data[ix]) for ix in ix_sequence])
        # return "[" + contents + "]"

    #####################
    # Note: start with the
    # simple stuff. __getitem__,
    # __len__, __init__, __str__, etc.

    def is_empty(self):
        """Returns true if no elements in queue"""
        return self._n == 0

    def dequeue(self):
        """Pop an item from the front of the queue (inch the front pointer along)"""
        if (self.is_empty()):
            raise Empty("oops, empty queue")

        dafront = self._data[self._front]

        self._data[self._front] = None  # clean up
        self._front = (self._front + 1) % (
            len(self._data))  # update front pointer
        self._n -= 1

        # Really, we need to resize this thing to be smaller
        # when we remove stuff from it.
        # If it is too big by a quarter, chop it in half.
        if (self._n < len(self._data) // 4):
            self.resize(len(self._data) // 2)

        return dafront

    def enqueue(self, e):
        """Add an item to the back of the queue"""
        if (self._n == len(self._data)):
            self.resize(2 * self._n)

        insert_index = (self._front + self._n) % (len(self._data))
        self._data[insert_index] = e
        self._n += 1

    def resize(self, newsize):
        """Resize, and shift everything to the front"""
        old = self._data
        walk = self._front
        self._data = [None] * newsize
        for k in range(self._n):
            self._data[k] = old[walk]
            walk = (walk + 1) % len(old)
        self._front = 0

    def first(self):
        return self._data[self._first]


class LinkedBinaryTree:
    class Node:
        def __init__(self, data, left=None, right=None):
            self.data = data
            self.parent = None
            self.left = left
            if (self.left is not None):
                self.left.parent = self
            self.right = right
            if (self.right is not None):
                self.right.parent = self

    def __init__(self, root=None):
        self.root = root
        self.size = self.subtree_count(root)

    def __len__(self):
        return self.size

    def is_empty(self):
        return len(self) == 0

    def subtree_count(self, root):
        if (root is None):
            return 0
        else:
            left_count = self.subtree_count(root.left)
            right_count = self.subtree_count(root.right)
            return 1 + left_count + right_count

    def sum(self):
        return self.subtree_sum(self.root)

    def subtree_sum(self, root):
        if (root is None):
            return 0
        else:
            left_sum = self.subtree_sum(root.left)
            right_sum = self.subtree_sum(root.right)
            return root.data + left_sum + right_sum

    def height(self):
        return self.subtree_height(self.root)

    def subtree_height(self, root):
        if (root.left is None and root.right is None):
            return 0
        elif (root.left is None):
            return 1 + self.subtree_height(root.right)
        elif (root.right is None):
            return 1 + self.subtree_height(root.left)
        else:
            left_height = self.subtree_height(root.left)
            right_height = self.subtree_height(root.right)
            return 1 + max(left_height, right_height)

    def preorder(self):
        yield from self.subtree_preorder(self.root)

    def subtree_preorder(self, root):
        if (root is None):
            return
        else:
            yield root
            yield from self.subtree_preorder(root.left)
            yield from self.subtree_preorder(root.right)

    def postorder(self):
        yield from self.subtree_postorder(self.root)

    def subtree_postorder(self, root):
        if (root is None):
            return
        else:
            yield from self.subtree_postorder(root.left)
            yield from self.subtree_postorder(root.right)
            yield root

    def inorder(self):
        yield from self.subtree_inorder(self.root)

    def subtree_inorder(self, root):
        if (root is None):
            return
        else:
            yield from self.subtree_inorder(root.left)
            yield root
            yield from self.subtree_inorder(root.right)

    def breadth_first(self):
        if (self.is_empty()):
            return
        line = ArrayQueue.ArrayQueue()
        line.enqueue(self.root)
        while (line.is_empty() == False):
            curr_node = line.dequeue()
            yield curr_node
            if (curr_node.left is not None):
                line.enqueue(curr_node.left)
            if (curr_node.right is not None):
                line.enqueue(curr_node.right)

    def __iter__(self):
        for node in self.breadth_first():
            yield node.data


def create_expression_tree(prefix_exp_str):
    # expr_lst = prefix_exp_str.split(" ")
    expr_lst = prefix_exp_str

    # op = {'+': 2, '-': 2, '*': 2, '/': 2, 'a': 2, 'o': 2, 'n': 1}
    op = {11: 2, 12: 2, 13: 1, 14: 1}
    val = [3, 4, 5, 6, 7, 8, 9, 10]

    def create_expression_tree_helper(prefix_exp, start_pos):
        start_pos += 1
        try:
            elem = prefix_exp[start_pos]
        except:
            elem = random.choice(val)

        node = None
        size = 1

        if elem not in op:
            node = LinkedBinaryTree.Node(int(elem))
        else:
            expect_tokens = op[elem]
            left, left_size = create_expression_tree_helper(prefix_exp,
                                                            start_pos)
            if expect_tokens > 1:
                right, right_size = create_expression_tree_helper(prefix_exp,
                                                                  start_pos + left_size)
            else:
                right, right_size = None, 0

            node = LinkedBinaryTree.Node(elem, left, right)
            size += left_size + right_size

        return node, size

    tree = LinkedBinaryTree(create_expression_tree_helper(expr_lst, -1)[0])

    return tree
