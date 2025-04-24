# A heavily modified version of NLTK's generator from
# nltk.parse.generate
#
# This file adds some quality of life changes as well
# as some additional functionality such as generating
# random sentences.

# Natural Language Toolkit: Generating from a CFG
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@heatherleaf.se>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

import itertools
import sys
import random
import nltk

from nltk.grammar import Nonterminal
from nltk.sem.logic import read_logic

def fast_generate(g, depth, n=None):

    strings = list(nltk.parse.generate.generate(g, depth=depth, n=n))
    expressions = set()
    for string in strings:

        expr = read_logic(" ".join(string))[0]
        expressions.add(expr)

    return expressions

# A new and iterative method for generating strings from a CFG
# Compare with generate() from nltk.parse.generate.generate().
#
# It is a bit slower at the moment but is much better at generating
# larger sequences and also honoring the CFG depth correctly.
#
# For the old generate, the depth parameter does not necessarily
# reflect the true depth of the grammar.
#
#
def generate(g, depth, n=None):

    depth_dict = {0: [None]}
    results_dict = {}
    for curr_depth in range(1, depth + 1):

        prev_strings = depth_dict[curr_depth - 1]

        if n:
            prev_strings = random.sample(prev_strings, min(len(prev_strings), n))

        for string in prev_strings:
            terminals, nt_list = _fast_generate(g, start=string, depth=1, n=n)

            depth_strings = depth_dict.setdefault(curr_depth, [])
            depth_strings += nt_list

            for t in terminals:

                results_set = results_dict.setdefault(curr_depth, [])
                results_set.append(t)

    return results_dict

def _fast_generate(grammar, start=None, depth=1, n=None):
    """
    Generates an iterator of all sentences from a CFG.

    :param grammar: The Grammar used to generate sentences.
    :param start: The Nonterminal from which to start generate sentences.
    :param depth: The maximal depth of the generated tree.
    :param n: The maximum number of sentences to return.
    :return: An iterator of lists of terminal tokens.
    """
    if not start:
        start = grammar.start()
    if depth is None:
        depth = sys.maxsize

    nt_list = []
    if not isinstance(start, list):
        iter = _generate_all(grammar, [start], [], depth,  nt_list)
    else:
        iter = _generate_all(grammar, start, [], depth, nt_list)

    if n:
        iter = list(itertools.islice(iter, n))
        nt_list = random.sample(nt_list, min(len(nt_list), n))
    else:
        iter = list(iter)

    return iter, nt_list


def _generate_all(grammar, items, f1, depth, nt_list):
    if items:
        try:
            for frag1 in _generate_one(grammar, items[0], f1, items, depth, nt_list):
                for frag2 in _generate_all(grammar, items[1:], f1 + frag1, depth, nt_list):
                    yield frag1 + frag2
        except RecursionError as error:
            # Helpful error message while still showing the recursion stack.
            raise RuntimeError(
                "The grammar has rule(s) that yield infinite recursion!"
            ) from error
    else:
        yield []


def _generate_one(grammar, item, f1, items, depth, nt_list):

    if isinstance(item, Nonterminal):
        if depth > 0:

            prods = grammar.productions(lhs=item)
            prods = random.sample(prods, len(prods))

            for prod in prods:
                for rule in prod.rhs():
                    if isinstance(rule, Nonterminal):
                        nt_list.append(f1 + list(prod.rhs()) + items[1:])
                        break

                yield from _generate_all(grammar, prod.rhs(), f1, depth - 1, nt_list)
    else:
        yield [item]

