# Semiring (unused), 
#   Log Semiring (unused),
#   Max Semiring (unused),
#   KMax Semiring (unused),
#   Entropy Semiring (unused),
#   Cross Entropy Semiring (unused),
#   KLDivergence Semiring (unused),
#   Sampled Semiring (unused),
#   Sparsemax Semiring (unused)
# Yu Zhang (unedited)
# UCT Honours Project: 
#   Meaning Representation Parsing: 
#   The Edge Prediction Component of a Semantic Graph Parser
#   by Claudia Greenberg, GRNCLA009
# September 2022
# Not directly used in this adaptation
# -*- coding: utf-8 -*-

from functools import reduce

from typing import Iterable

import torch

from supar.utils.common import MIN
from supar.structs.fn import sampled_logsumexp, sparsemax


class Semiring(object):
    r"""
    Base semiring class :cite:`goodman-1999-semiring`.

    A semiring is defined by a tuple :math:`<K, \oplus, \otimes, \mathbf{0}, \mathbf{1}>`.
    :math:`K` is a set of values;
    :math:`\oplus` is commutative, associative and has an identity element `0`;
    :math:`\otimes` is associative, has an identity element `1` and distributes over `+`.
    """

    zero = 0
    one = 1

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x.sum(dim)

    @classmethod
    def add(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cls.sum(torch.stack((x, y)), 0)

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y

    @classmethod
    def dot(cls, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return cls.sum(cls.mul(x, y), dim)

    @classmethod
    def prod(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x.prod(dim)

    @classmethod
    def times(cls, *x: Iterable[torch.Tensor]) -> torch.Tensor:
        return reduce(lambda i, j: cls.mul(i, j), x)

    @classmethod
    def zero_(cls, x: torch.Tensor) -> torch.Tensor:
        return x.fill_(cls.zero)

    @classmethod
    def one_(cls, x: torch.Tensor) -> torch.Tensor:
        return x.fill_(cls.one)

    @classmethod
    def zero_mask(cls, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return x.masked_fill(mask, cls.zero)

    @classmethod
    def zero_mask_(cls, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return x.masked_fill_(mask, cls.zero)

    @classmethod
    def one_mask(cls, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return x.masked_fill(mask, cls.one)

    @classmethod
    def one_mask_(cls, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return x.masked_fill_(mask, cls.one)

    @classmethod
    def zeros_like(cls, x: torch.Tensor) -> torch.Tensor:
        return x.new_full(x.shape, cls.zero)

    @classmethod
    def ones_like(cls, x: torch.Tensor) -> torch.Tensor:
        return x.new_full(x.shape, cls.one)

    @classmethod
    def convert(cls, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def unconvert(cls, x: torch.Tensor) -> torch.Tensor:
        return x


class LogSemiring(Semiring):
    r"""
    Log-space semiring :math:`<\mathrm{logsumexp}, +, -\infty, 0>`.
    """

    zero = MIN
    one = 0

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x.logsumexp(dim)

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    @classmethod
    def prod(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x.sum(dim)


class MaxSemiring(LogSemiring):
    r"""
    Max semiring :math:`<\mathrm{max}, +, -\infty, 0>`.
    """

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x.max(dim)[0]


def KMaxSemiring(k):
    r"""
    k-max semiring :math:`<\mathrm{kmax}, +, [-\infty, -\infty, \dots], [0, -\infty, \dots]>`.
    """

    class KMaxSemiring(LogSemiring):

        @classmethod
        def convert(cls, x: torch.Tensor) -> torch.Tensor:
            return torch.cat((x.unsqueeze(-1), cls.zero_(x.new_empty(*x.shape, k - 1))), -1)

        @classmethod
        def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
            return x.movedim(dim, -1).flatten(-2).topk(k, -1)[0]

        @classmethod
        def mul(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return (x.unsqueeze(-1) + y.unsqueeze(-2)).flatten(-2).topk(k, -1)[0]

        @classmethod
        def one_(cls, x: torch.Tensor) -> torch.Tensor:
            x[..., :1].fill_(cls.one)
            x[..., 1:].fill_(cls.zero)
            return x

    return KMaxSemiring


class EntropySemiring(LogSemiring):
    r"""
    Entropy expectation semiring :math:`<\oplus, +, [-\infty, 0], [0, 0]>`,
    where :math:`\oplus` computes the log-values and the running distributional entropy :math:`H[p]`
    :cite:`li-eisner-2009-first,hwa-2000-sample,kim-etal-2019-unsupervised`.
    """

    @classmethod
    def convert(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x, cls.ones_like(x)), -1)

    @classmethod
    def unconvert(cls, x: torch.Tensor) -> torch.Tensor:
        return x[..., -1]

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        p = x[..., 0].logsumexp(dim)
        r = x[..., 0] - p.unsqueeze(dim)
        r = r.exp().mul((x[..., -1] - r)).sum(dim)
        return torch.stack((p, r), -1)

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    @classmethod
    def zero_(cls, x: torch.Tensor) -> torch.Tensor:
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x: torch.Tensor) -> torch.Tensor:
        return x.fill_(cls.one)


class CrossEntropySemiring(LogSemiring):
    r"""
    Cross Entropy expectation semiring :math:`<\oplus, +, [-\infty, -\infty, 0], [0, 0, 0]>`,
    where :math:`\oplus` computes the log-values and the running distributional cross entropy :math:`H[p,q]`
    of the two distributions :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, cls.one_(torch.empty_like(x[..., :1]))), -1)

    @classmethod
    def unconvert(cls, x: torch.Tensor) -> torch.Tensor:
        return x[..., -1]

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        p = x[..., :-1].logsumexp(dim)
        r = x[..., :-1] - p.unsqueeze(dim)
        r = r[..., 0].exp().mul((x[..., -1] - r[..., 1])).sum(dim)
        return torch.cat((p, r.unsqueeze(-1)), -1)

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    @classmethod
    def zero_(cls, x: torch.Tensor) -> torch.Tensor:
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x: torch.Tensor) -> torch.Tensor:
        return x.fill_(cls.one)


class KLDivergenceSemiring(LogSemiring):
    r"""
    KL divergence expectation semiring :math:`<\oplus, +, [-\infty, -\infty, 0], [0, 0, 0]>`,
    where :math:`\oplus` computes the log-values and the running distributional KL divergence :math:`KL[p \parallel q]`
    of the two distributions :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, cls.one_(torch.empty_like(x[..., :1]))), -1)

    @classmethod
    def unconvert(cls, x: torch.Tensor) -> torch.Tensor:
        return x[..., -1]

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        p = x[..., :-1].logsumexp(dim)
        r = x[..., :-1] - p.unsqueeze(dim)
        r = r[..., 0].exp().mul((x[..., -1] - r[..., 1] + r[..., 0])).sum(dim)
        return torch.cat((p, r.unsqueeze(-1)), -1)

    @classmethod
    def mul(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    @classmethod
    def zero_(cls, x: torch.Tensor) -> torch.Tensor:
        x[..., :-1].fill_(cls.zero)
        x[..., -1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x: torch.Tensor) -> torch.Tensor:
        return x.fill_(cls.one)


class SampledSemiring(LogSemiring):
    r"""
    Sampling semiring :math:`<\mathrm{logsumexp}, +, -\infty, 0>`,
    which is an exact forward-filtering, backward-sampling approach.
    """

    @classmethod
    def sum(cls, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sampled_logsumexp(x, dim)


class SparsemaxSemiring(LogSemiring):
    r"""
    Sparsemax semiring :math:`<\mathrm{sparsemax}, +, -\infty, 0>`
    :cite:`martins-etal-2016-sparsemax,mensch-etal-2018-dp,correia-etal-2020-efficient`.
    """

    @staticmethod
    def sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        p = sparsemax(x, dim)
        return x.mul(p).sum(dim) - p.norm(p=2, dim=dim)
