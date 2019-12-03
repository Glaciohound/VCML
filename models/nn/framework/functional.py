#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : functional.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 23.07.2019
# Last Modified Date: 23.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
# Several functional classes

import torch
import torch.nn as nn
import torch.autograd as autograd
import math

from utility.common import \
    assert_valid_value
from .sub_functional import ln_pdf, ln_cdf, logit_ln
from .utils import infinite  # , clamp_infinite


class HalfGaussianConditionalLogit(nn.Module):
    """
    Calculating the logits for a HalfGaussianIntersection
    conditional probability given two supporting vectors.

    Input:
        x: Tensor, representing a set of vectors
        y: Tensor, representing a set of vectors

    Output:
        logit: Tensor, logit(Pr(y | x))

    """

    def __init__(self, n_sample, max_value, device, slack=False):
        super().__init__()
        self.n_sample = n_sample
        self.max_value = max_value
        self.slack = slack
        self.device = device
        self.ln_intersection_fn = get_LnIntersection(
            self.n_sample, self.max_value, self.device, self.slack
        )

    def forward(self, x_vec, y_vec):
        if x_vec.dim() == 1:
            return self(x_vec[None], y_vec)[0]
        elif y_vec.dim() == 1:
            return self(x_vec, y_vec[None])[:, 0]

        ln_conditional, _, _, _ = self.ln_conditional(x_vec, y_vec)
        logit_conditional = logit_ln(ln_conditional, self.slack)

        return logit_conditional

    def ln_conditional(self, x_vec, y_vec):
        x = x_vec.norm(2, -1)
        y = y_vec.norm(2, -1)
        equal = (x_vec[:, None] - y_vec[None]).abs().sum(2) == 0
        inner_prod = torch.matmul(x_vec, y_vec.t())
        cos = inner_prod / x[:, None] / y[None]
        cos[equal] = 1

        ln_intersection = self.ln_intersection_fn(
            x, y, cos,
        )
        ln_x = ln_cdf_by_integral(
            -x, self.ln_intersection_fn)[:, None]
        ln_conditional = ln_intersection - ln_x
        return ln_conditional, x, y, cos

    def ln_lambda(self, x_vec, y_vec):
        ln_conditional, _, y, _, = self.ln_conditional(x_vec, y_vec)
        ln_lambda = ln_conditional - ln_cdf_by_integral(
            -y, self.ln_intersection_fn)[None]
        return ln_lambda


def get_LnIntersection(n_sample, max_value, device, slack):

    class LnIntersection(autograd.Function):
        """
        Calculating the ln probability defined by Pr(X ∩ Y),
        given the support vectors of X, Y and their inter-cosine values

        ln(Pr) = ln(∫[x, +∞](ø(t)·Ø(-g(u, y))))
        g(u, y) = (y - x·cos(theta)) / sin(theta)

        Input:
            x: Tensor, shape = (n_x, dim)
            y: Tensor, shape = (n_y, dim)
            cos: Tensor, shape = (n_x, n_y)

        Output:
            ln(Pr), Tensor, shape = (n_x, n_y)
        """

        current_device = torch.cuda.current_device()
        points = torch.linspace(0, max_value, n_sample).to(device)
        c = ln_pdf(points)[None, None, :]

        @classmethod
        def forward_inner(cls, x, y, cos):
            points = cls.points
            c = cls.c
            delta = max_value / (n_sample - 1)
            sin = (1 - cos.pow(2)).clamp(0, 1).sqrt()
            csc = (1 / sin).clamp(0, infinite)

            # re-order x and y, letting x be the bigger of the two
            x_r = torch.max(x[:, None], y[None])
            y_r = torch.min(x[:, None], y[None])

            a = ln_pdf(x_r)
            b = ln_cdf(-(y_r - x_r * cos) * csc, True, slack)
            g_uy = (y_r[:, :, None] -
                    points[None, None, :] * cos[:, :, None]) * csc[:, :, None]
            d = ln_cdf(-g_uy, True, slack)
            e = (c + d - (a + b)[:, :, None]).exp()
            out = points[None, None, :] <= x_r[:, :, None]
            e[out] = 0

            ln_Pr = a + b + e.sum(2).log() + math.log(delta)

            return ln_Pr, a, b, c, d, e, g_uy, sin, csc

        @staticmethod
        def forward(ctx, x, y, cos):
            """
            The forward calculation is approximated as :
            ln(Pr) = ln(ø(x)·Ø(-g(x, y))) +
                     ln([x, +∞]∫(ø(u)Ø(-g(u, y)) / ø(x)Ø(-g(x, y))))
            noted as output = a + b + ln(sum(exp(c + d - a - b)))

            In calculation, the index dimensions are in order of [x, y, u]
            """

            ln_Pr, a, b, c, d, e, g_uy, sin, csc = \
                LnIntersection.forward_inner(
                    x, y, cos
                )

            ctx.save_for_backward(x, y, cos, sin, csc, ln_Pr,
                                  torch.BoolTensor([slack]))
            if not slack:
                try:
                    assert_valid_value(ln_Pr)
                except Exception:
                    from pprint import pprint
                    pprint((a, b, c, d, e, g_uy, ln_Pr, x, y, cos, sin, csc))
                    raise Exception()

            return ln_Pr

        @staticmethod
        def backward(ctx, grad_output):
            """
            grad(ln(Pr)) = grad(Pr) / Pr

            For numerical stability, the log of the negative gradient is
            calculated first, and taken exponent later e.g. grad[x](ln(Pr)) = -
            exp(ln(grad[x](Pr)) - ln(Pr)) * grad_output
            """
            x, y, cos, sin, csc, ln_Pr, slack = ctx.saved_tensors
            x = x[:, None]
            y = y[None]

            # Calculating ln_grad(ln(Pr)
            ln_grad_x = ln_cdf((x * cos - y) * csc, True, slack) +\
                ln_pdf(x) - ln_Pr
            ln_grad_y = ln_cdf((y * cos - x) * csc, True, slack) +\
                ln_pdf(y) - ln_Pr
            ln_grad_cos = \
                - (x.pow(2) - 2 * x * y * cos + y.pow(2)) * csc * csc / 2 \
                + (csc / 2 / math.pi).log() - ln_Pr

            # grad(ln(Pr))
            grad_x = - ln_grad_x.exp() * grad_output
            grad_y = - ln_grad_y.exp() * grad_output
            grad_cos = ln_grad_cos.exp() * grad_output

            # summing gradient flows
            grad_x = grad_x.sum(1)
            grad_y = grad_y.sum(0)

            # clamping value
            # grad_cos = clamp_infinite(grad_cos)
            grad_cos[sin == 0] = 0

            if not slack:
                assert_valid_value(grad_x, grad_y, grad_cos,
                                   assert_finite=True)
            return grad_x, grad_y, grad_cos

    ln_intersection_fn = LnIntersection().apply

    return ln_intersection_fn


def ln_cdf_by_integral(mx, ln_intersection_fn):
    """
    Calculate the ln_cdf(x) by integration.
    This function is not as accurate as ln_cdf, but has no gap between the
    value of ln_intersection_fn, solving the underfitting problem in visual
    classification.
    """

    assert (mx < 0).all()
    x = -mx
    y = x.min()[None] / 2
    n_x = x.shape[0]
    output = ln_intersection_fn(
        x, y, torch.ones(n_x, 1).to(x.device)
    )[:, 0]
    return output
