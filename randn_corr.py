#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
# Generate N-dim normal distribution given mu and corr
# Date: 2016/08/06
# Author: Gilbert Liou (gilbert.liou.tw@gmail.com)


def genrand(mu, corr, T):
    n = len(mu)
    mat = np.random.randn(n, T)
    chol = np.linalg.cholesky(corr)
    mat = np.broadcast_to(mu, [T, n]).T + np.dot(chol, mat)
    return mat


if __name__ == '__main__':
    n, T = 3, 10000
    mu = [0.2, 0.3, -0.1]
    corr = 0.3 * np.ones((n, n), dtype=float)
    for i in range(n):
        corr[i, i] = 1.

    print '=== True value ==='
    print 'mu'
    print mu
    print 'corr'
    print corr

    mat = genrand(mu, corr, T)
    print '=== Sample ==='
    print 'mu'
    print np.mean(mat, axis=1)
    print 'corr'
    print np.corrcoef(mat)
