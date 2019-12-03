#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : sub_functional.py
# Author            : Chi Han, Jiayuan Mao
# Email             : haanchi@gmail.com, maojiayuan@gmail.com
# Date              : 29.07.2019
# Last Modified Date: 23.10.2019
# Last Modified By  : Chi Han, Jiayuan Mao
#
# This file is part of the VCML codebase
# Distributed under MIT license
# Several functional classes

import math
import torch
import torch.autograd as autograd
from torch.distributions import Normal
import numpy as np

from utility.common import assert_valid_value
from .utils import \
    infinitesimal, infinite, clamp_infinite


class LogitLn_cls(autograd.Function):
    """
    Logit function, taking in the ln of a probability as x,
    and outputs the logits of it.
    This implementation is numerically stable.
    ln (p / (1-p)), p = exp(x)

    Input:
        x: Tensor, x < 0, the ln of the probability

    Output:
        y: Tensor, the logit of the probability
    """

    @staticmethod
    def forward(ctx, x, slack=False):
        """
        logit = x - ln(1 - exp(x))
        When x is close to 0, exp(x) == 1,
        the second term will be approximated by ln(-x).

        So the final calculation is:
        logit = a - b
        b = { b1, if exp(x) != 1
            | b2, otherwise

        Before calculation, x is cast below -infinitesimal
        """
        clamped = x > -infinitesimal
        x = x.clamp(-infinite, -infinitesimal)
        ctx.save_for_backward(x, clamped, torch.BoolTensor([slack]))

        a = x
        b = stable_softminus(x)
        logit = a - b + (x + infinitesimal) * clamped.float()

        if not slack:
            assert_valid_value(logit)
        return logit

    @staticmethod
    def backward(ctx, grad_output):
        """
        d(logit) / d(x) = 1 + 1 / (exp(-x) - 1)
        Similarly, if x is close to 1, exp(-x) == 1,
        the second term will be approximated by -(1 / x)

        grad_x = grad_output * (1 + 1 / a)
        a = { exp(-x) - 1, if exp(x) != 1)
            | -x, otherwise
        """

        x, clamped, slack = ctx.saved_tensors

        a1 = (-x).exp() - 1
        a2 = -x
        switch = x.exp() == 1
        a1[switch] = 0
        a2[~switch] = 0
        a = a1 + a2

        grad_x = grad_output * (1 + 1 / a)
        # clip out gradient when x is super high
        grad_x[clamped] = 1

        if not slack:
            assert_valid_value(grad_x)
        return grad_x, None


class ln_cdf_cls(autograd.Function):
    """
    Calculating the log of normal-cdf function

    Input:
        x: Tensor

    Output:
        ln: ln of cdf function
    """

    @staticmethod
    def forward(ctx, x, recursive=True, slack=False):
        """
        Method 1 follows that of log_ndtr in scipy:
            https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c
        Method 2 follows from "Abramowitz and Stegun", 7.1.28
        Method 3 follows from "global Padé approximations"
        """

        method_chosen = 2

        if method_chosen == 1:
            # Method 1
            normal = Normal(0, 1)
            th1 = 5
            th2 = -5
            switch1 = x >= th1
            switch2 = (x >= th2) * (x < th1)
            switch3 = x < th2

            if recursive:
                ln1 = -cdf(-x, False)
            else:
                ln1 = -normal.cdf(-x)

            ln2 = normal.cdf(x).log()

            ln3_1 = - x.pow(2) / 2 - (-x).log() - math.log(2*math.pi) / 2
            ln3_2 = 1
            for i in reversed(range(1, 6)):
                ln3_2 = 1 - (2 * i - 1) * x.pow(-2) * ln3_2
            ln3_2 = ln3_2.log()
            ln3 = ln3_1 + ln3_2

            ln1[~switch1] = 0
            ln2[~switch2] = 0
            ln3[~switch3] = 0

            ln = ln1 + ln2 + ln3

        elif method_chosen == 2:
            # Method 2
            p = 0.3275911
            a = np.array([0,
                          0.254829592, -0.284496736,
                          1.421413741, -1.453152027, 1.061405429])
            t = 1 / (1 + p * x.abs() / math.sqrt(2))
            poly = torch.stack([a[i] * t.pow(i) for i in range(1, 6)]).sum(0)
            ln_inside = clamp_infinite(
                -x.pow(2)/2 + poly.log() - math.log(2))

            ln1 = stable_softminus(ln_inside)
            ln2 = ln_inside
            ln1[x < 0] = 0
            ln2[x >= 0] = 0
            ln = ln1 + ln2

        elif method_chosen == 3:
            # Method 3
            a = 0.140012
            g = x.pow(2)/2 * ((4/math.pi + a*x.pow(2)/2) / (1 + a*x.pow(2)/2))
            ln_eg4 = - g - math.log(4)

            # positive normal range
            ln1 = ((1 + (1 - (-g).exp()).sqrt()) / 2).log()
            # positive infinity
            # ln2 = stable_softminus(ln_eg4 + F.softplus(ln_eg4))
            ln2 = - (ln_eg4).exp() - (2*ln_eg4).exp()*3/2 \
                - (3*ln_eg4).exp()*10/3
            # negaitve normal range
            ln3 = ((1 - (1 - (-g).exp()).sqrt()) / 2).log()
            # negative infinity
            ln4 = ln_eg4 - ln2

            infinity_flag = (1 - (-g).exp()).sqrt() == 1
            switch1 = (x >= 0) * (~infinity_flag)
            switch2 = (x >= 0) * infinity_flag
            switch3 = (x < 0) * (~infinity_flag)
            switch4 = (x < 0) * infinity_flag
            ln1[~switch1] = 0
            ln2[~switch2] = 0
            ln3[~switch3] = 0
            ln4[~switch4] = 0

            ln = ln1 + ln2 + ln3 + ln4

        if not slack:
            assert_valid_value(ln)
        ctx.save_for_backward(x, ln, torch.BoolTensor([slack]))
        return ln

    @staticmethod
    def backward(ctx, grad_output):
        x, ln_cdf_x, slack = ctx.saved_tensors
        ln_pdf_x = ln_pdf(x)
        grad_x = (ln_pdf_x - ln_cdf_x).exp() * grad_output
        if not slack:
            assert_valid_value(grad_x, assert_finite=True)
        return grad_x, None


class ln_pdf_cls(autograd.Function):
    """
    Calculating the log of normal-pdf function

    Input:
        x: Tensor

    Output:
        ln: ln of cdf function
    """

    @staticmethod
    def forward(ctx, x, slack=False):
        ln = - x.pow(2) / 2 - math.log(2 * math.pi) / 2

        ctx.save_for_backward(x, torch.BoolTensor([slack]))
        if not slack:
            assert_valid_value(ln)
        return ln

    @staticmethod
    def backward(ctx, grad_output):
        x, slack = ctx.saved_tensors

        grad_x = -x * grad_output
        if not slack:
            assert_valid_value(grad_x)
        return grad_x, None


ln_pdf = ln_pdf_cls().apply
ln_cdf = ln_cdf_cls().apply
logit_ln = LogitLn_cls().apply


def pdf(x):
    output = ln_pdf(x).exp()
    return output


def cdf(x, recursive=True):
    output = ln_cdf(x, recursive).exp()
    return output


def stable_softminus(x):
    """
    Calculating log(1 - exp(x)) (x < 0) in a numerically stable way
    """
    # negative normal range
    y1 = (1 - x.exp()).log()
    # negative zero
    y2 = (- x - x.pow(2)/2 - x.pow(3)/6).log()
    # negative infinity
    y3 = - x.exp() - (2*x).exp()/2 - (3*x).exp()/3
    switch2 = (1 - x.exp()) == 0
    switch3 = (1 - x.exp()) == 1
    switch1 = ~switch2 ^ switch3

    y1[~switch1] = 0
    y2[~switch2] = 0
    y3[~switch3] = 0
    y = y1 + y2 + y3

    return y


__all__ = ['ln_pdf', 'ln_cdf', 'pdf', 'cdf', 'logit_ln']
