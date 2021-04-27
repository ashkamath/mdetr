# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Simple union find structure implementation"""


class UnionFind:
    """Optimized union find structure"""

    def __init__(self, n):
        """Initialize a union find with n components"""
        self.compo = list(range(n))
        self.weight = [1] * n
        self.nb_compo = n

    def get_nb_compo(self):
        return self.nb_compo

    def find(self, x):
        if self.compo[x] == x:
            return x
        self.compo[x] = self.find(self.compo[x])
        return self.compo[x]

    def unite(self, a, b):
        fa = self.find(a)
        fb = self.find(b)
        if fa != fb:
            self.nb_compo -= 1
            if self.weight[fb] > self.weight[fa]:
                fa, fb = fb, fa
            self.compo[fb] = fa
            self.weight[fa] += self.weight[fb]
