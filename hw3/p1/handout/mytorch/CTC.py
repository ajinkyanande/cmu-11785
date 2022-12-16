import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):

        """Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        self.BLANK = BLANK

    def extend_target_with_blank(self, target):

        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]

        for symbol in target:

            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        skip_connect = []

        for i in range(N):
            
            if i <= 2 or extended_symbols[i] == 0:
                skip_connect.append(0)
            
            if i > 2 and extended_symbols[i-2] != 0:
                skip_connect.append(1)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):

        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        T, S = len(logits), len(extended_symbols)
        alpha = np.zeros(shape=(T, S))

        alpha[0, 0] = logits[0, extended_symbols[0]]
        alpha[0, 1] = logits[0, extended_symbols[1]]
        alpha[0, 2:] = 0.0

        for t in range(1, T):
            
            alpha[t, 0] = alpha[t-1, 0] * logits[t, extended_symbols[0]]
            
            for sym in range(1, S):
                alpha[t, sym] = alpha[t-1, sym] + alpha[t-1, sym-1]

                if sym > 1 and extended_symbols[sym] != extended_symbols[sym-2]:
                    alpha[t, sym] += alpha[t-1, sym-2]
                
                alpha[t, sym] *= logits[t, extended_symbols[sym]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        T, S = len(logits), len(extended_symbols)
        beta = np.zeros(shape=(T, S))

        beta[T-1, S-1] = logits[T-1, extended_symbols[S-1]]
        beta[T-1, S-2] = logits[T-1, extended_symbols[S-2]]
        beta[T-1, 0:S-3] = 0.0

        for t in reversed(range(T-1)):

            beta[t, S-1] = beta[t+1, S-1] * logits[t, extended_symbols[S-1]]

            for sym in reversed(range(S-1)):
            
                beta[t, sym] = beta[t+1, sym] + beta[t+1, sym+1]

                if sym <= S-3 and extended_symbols[sym] != extended_symbols[sym+2]:
                    beta[t, sym] += beta[t+1, sym+2]
                
                beta[t, sym] *= logits[t, extended_symbols[sym]]

        for t in reversed(range(T)):
            
            for sym in reversed(range(S)):

                beta[t, sym] = beta[t, sym] / logits[t, extended_symbols[sym]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        (T, S) = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        for t in range(T):

            sumgamma[t] = 0.0

            for sym in range(S):

                gamma[t, sym] = alpha[t, sym] * beta[t, sym]
                sumgamma[t] += gamma[t, sym]

            for sym in range(S):
                
                gamma[t, sym] = gamma[t, sym] / sumgamma[t]

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        
        """Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """

        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    def __call__(self, logits, target, input_lengths, target_lengths):

        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        (B, _) = self.target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):

            one_logits = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            one_target = self.target[batch_itr, :self.target_lengths[batch_itr]]

            extended_target, skip_connect_target = self.ctc.extend_target_with_blank(one_target)

            alpha = self.ctc.get_forward_probs(one_logits, extended_target, skip_connect_target)
            beta = self.ctc.get_backward_probs(one_logits, extended_target, skip_connect_target)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            for i in range(gamma.shape[0]):

                for j in range(gamma.shape[1]):
                
                    total_loss[batch_itr] += - gamma[i, j] * np.log(one_logits[i, extended_target[j]])

        total_loss = np.sum(total_loss) / B
        
        return total_loss
    
    def backward(self):

        """CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        (B, _) = self.target.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):

            one_logits = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            one_target = self.target[batch_itr, :self.target_lengths[batch_itr]]

            extended_target, skip_connect_target = self.ctc.extend_target_with_blank(one_target)

            alpha = self.ctc.get_forward_probs(one_logits, extended_target, skip_connect_target)
            beta = self.ctc.get_backward_probs(one_logits, extended_target, skip_connect_target)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            for i in range(gamma.shape[0]):

                for j in range(gamma.shape[1]):
                
                    dY[i, batch_itr, extended_target[j]] += - gamma[i, j] / one_logits[i, extended_target[j]]

        return dY