#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy


class HMM:
    def __init__(self, states, transition, emission, init):
        self.state_names = copy.copy(states)
        self.n_states = len(states)
        self.A = transition.copy()
        self.B = emission.copy()
        self.n_emissions = self.B.shape[1]
        self.init = init

    def generate(self, length):
        state = self.init
        states = []
        ret = []
        for i in xrange(1, length + 1):
            state = np.random.choice(range(self.n_states), p=self.A[state])
            states.append(state)
            ret.append(
                np.random.choice(range(self.n_emissions), p=self.B[state]))
        print(''.join(self.state_names[i] for i in states))
        ret = ''.join([str(i) for i in ret])
        return ret

    def _forward(self, seq_arr):
        T = len(seq_arr)
        alpha = np.zeros((T + 1, self.n_states))
        alpha[0, self.init] = 1
        log_px = 0.
        for t in xrange(1, T + 1):
            alpha[t] = self.B[:, seq_arr[t - 1]] * \
                       np.dot(alpha[t - 1], self.A)
            pt = alpha[t].sum()
            alpha[t] /= pt
            log_px += np.log(pt)
        return alpha, log_px

    def _backward(self, seq_arr):
        T = len(seq_arr)
        beta = np.zeros((T + 1, self.n_states))
        beta[T, :] = 1
        log_px = 0.
        for t in xrange(T, 0, -1):
            beta[t - 1] = np.dot(self.A, beta[t] * self.B[:, seq_arr[t - 1]])
            pt = beta[t - 1].sum()
            beta[t - 1] /= pt
            log_px += np.log(pt)
        log_px += np.log(beta[0, self.init])
        return beta, log_px

    def viterbi(self, seq):
        # := max-product
        seq_arr = np.array([int(i) for i in seq])
        T = len(seq_arr)
        T1 = np.zeros((self.n_states, T + 1))
        T1[self.init, 0] = 1
        T2 = np.zeros((self.n_states, T + 1), dtype='int')
        states = np.zeros(T + 1, dtype='int')
        for t in xrange(1, T + 1):
            for j in xrange(self.n_states):
                T1[j, t] = np.max(T1[:, t - 1] * self.A[:, j])
                T1[j, t] *= self.B[j, seq_arr[t - 1]]
                T2[j, t] = np.argmax(T1[:, t - 1] * self.A[:, j])
        states[T] = np.argmax(T1[:, T])
        for t in xrange(T, 1, -1):
            states[t - 1] = T2[states[t], t - 1]
        return ''.join([self.state_names[s] for s in states[1:]])

    def baum_welch(self, seq):
        # := EM
        seq_arr = np.array([int(i) for i in seq])
        T = len(seq_arr)
        kesi = np.zeros((T + 1, self.n_states, self.n_states))
        log_px = None
        iter = 0
        while True:
            iter += 1
            alpha, alpha_log_px = self._forward(seq_arr)
            print "Iter %d" % iter, "log p(x): %s" % alpha_log_px
            if log_px and (np.abs(
                    log_px - alpha_log_px) < np.abs(1e-6 * log_px)):
                print "Converged."
                break
            beta, beta_log_px = self._backward(seq_arr)
            try:
                assert np.abs(
                    alpha_log_px - beta_log_px) < np.abs(1e-6 * alpha_log_px)
            except AssertionError as e:
                print "alpha_log_px:", alpha_log_px
                print "beta_log_px:", beta_log_px
                raise e
            log_px = alpha_log_px
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            for t in xrange(1, T):
                kesi[t] = np.outer(
                    alpha[t],
                    beta[t + 1] * self.B[:, seq_arr[t + 1 - 1]]) * self.A
            kesi[1:T] = kesi[1:T] / kesi[1:T].sum(axis=(1, 2), keepdims=True)
            self.A = kesi[1:T].sum(axis=0) / \
                     gamma[1:T].sum(axis=0)[:, np.newaxis]
            assert np.all(np.abs(1. - self.A.sum(axis=1)) < 1e-6)
            obs = np.zeros((T + 1, self.n_emissions))
            obs[range(1, T + 1), seq_arr] = 1
            self.B = np.dot(gamma[1:].T, obs[1:]) / \
                     gamma[1:].sum(axis=0)[:, np.newaxis]
        print "Estimate A:"
        print np.array_str(self.A, precision=3)
        print "Estimate B:"
        print np.array_str(self.B, precision=3)
        return log_px, self.A, self.B

    def gibbs(self, seq, steps=1, burn_in=0, max_iters=None):
        seq_arr = np.array([int(i) for i in seq])
        T = len(seq_arr)
        states = np.zeros(T + 1, dtype='int')
        iter = 0
        log_px = None
        states[0] = self.init
        while True:
            iter += 1
            alpha, alpha_log_px = self._forward(seq_arr)
            print "Iter %d" % iter, "log p(x): %s" % alpha_log_px
            if log_px and (np.abs(
                    log_px - alpha_log_px) < np.abs(1e-6 * log_px)):
                print "Converged."
                break
            log_px = alpha_log_px
            if max_iters and (iter >= max_iters):
                break
            A = np.zeros_like(self.A)
            B = np.zeros_like(self.B)
            for t in xrange(1, T + 1):
                states[t] = np.random.choice(range(3))
            for step in xrange(steps):
                for t in xrange(1, T + 1):
                    p_state_t = self.B[:, seq_arr[t - 1]] * \
                                self.A[states[t - 1]]
                    if t < T:
                        p_state_t *= self.A[:, states[t + 1]]
                    p_state_t /= p_state_t.sum()
                    states[t] = np.random.choice(range(3), p=p_state_t)
                if step >= burn_in:
                    for t in xrange(1, T + 1):
                        if t < T:
                            A[states[t], states[t + 1]] += 1
                        B[states[t], seq_arr[t - 1]] += 1
            A = np.maximum(1., A)
            B = np.maximum(1., B)
            self.A = A / A.sum(axis=1, keepdims=True)
            self.B = B / B.sum(axis=1, keepdims=True)
        print "Estimate A:"
        print np.array_str(self.A, precision=3)
        print "Estimate B:"
        print np.array_str(self.B, precision=3)
        return log_px


if __name__ == "__main__":
    np.random.seed(1236)
    states = ['A', 'B', 'C']

    print "1.1 Generation\n"
    transition = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.7, 0.2],
        [0.1, 0.0, 0.9]
    ])
    emission = np.array([
        [0.9, 0.1],
        [0.5, 0.5],
        [0.1, 0.9]
    ])
    init = 0
    hmm = HMM(states, transition, emission, init)
    seqs = []
    for seq_len in [100, 1000, 10000]:
        seq = hmm.generate(seq_len)
        seqs.append(seq)
        print "Inferred optimal state series:"
        print hmm.viterbi(seq)
        # NOTE: To run chains with various length, REMOVE this break
        # break

    print "\n1.2/1.3 Baum Welch"
    for seq in seqs:
        print "\nSequence length:", len(seq)
        As = []
        Bs = []
        for run in xrange(10):
            print "Run", run
            transition2 = np.random.random((3, 3))
            transition2 /= transition2.sum(axis=1, keepdims=True)
            emission2 = np.random.random((3, 2))
            emission2 /= emission2.sum(axis=1, keepdims=True)
            print "Init transition:"
            print transition2
            print "Init emission:"
            print emission2
            hmm2 = HMM(states, transition2, emission2, init)
            log_px, A, B = hmm2.baum_welch(seq)
            As.append(A)
            Bs.append(B)
            print "Final log p(x):", log_px
            print "Optimal state series:", hmm2.viterbi(seq)
            # NOTE: To calculate variance, REMOVE this break
            # break
        print "Variance of estimated:"
        print "Var(A):"
        print np.var(As, axis=0)
        print "Var(B):"
        print np.var(Bs, axis=0)

    print "\n2.1 Gibbs"
    transition3 = np.random.random((3, 3))
    transition3 /= transition3.sum(axis=1, keepdims=True)
    emission3 = np.random.random((3, 2))
    emission3 /= emission3.sum(axis=1, keepdims=True)
    print "Init transition:"
    print transition3
    print "Init emission:"
    print emission3
    print "Sequence length:", len(seqs[0])
    hmm3 = HMM(states, transition3, emission3, init)
    log_px = hmm3.gibbs(seqs[0], steps=200, burn_in=100, max_iters=100)
    print "Final log p(x):", log_px
    print "Optimal state series:", hmm3.viterbi(seqs[0])
