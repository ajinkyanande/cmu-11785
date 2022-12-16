import numpy as np
from activation import *


class GRUCell(object):

    def __init__(self, in_dim, hidden_dim):

        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.n_act = Tanh()

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):

        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):

        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):

        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """

        self.x = x
        self.hidden = h_prev_t
        
        # self.r = self.r_act.forward(x @ self.Wrx.T + self.brx + self.hidden @ self.Wrh.T + self.brh)
        # self.z = self.z_act.forward(x @ self.Wzx.T + self.bzx + self.hidden @ self.Wzh.T + self.bzh)
        # self.n = self.n_act.forward(x @ self.Wnx.T + self.bnx + self.r * (self.hidden @ self.Wnh.T + self.bnh))
        
        # h_t = (1 - self.z) * self.n + self.z * h_prev_t

        self.z1 = self.x @ self.Wrx.T # 1.1
        self.z2 = self.z1 + self.brx # 1.2
        self.z3 = self.hidden @ self.Wrh.T # 1.3
        self.z4 = self.z3 + self.brh # 1.4
        self.z5 = self.z2 + self.z4 # 1.5
        self.r = self.r_act.forward(self.z5) # 1.6

        self.z6 = self.x @ self.Wzx.T # 2.1
        self.z7 = self.z6 + self.bzx # 2.2
        self.z8 = self.hidden @ self.Wzh.T # 2.3
        self.z9 = self.z8 + self.bzh # 2.4
        self.z10 = self.z7 + self.z9 # 2.5
        self.z = self.z_act.forward(self.z10) # 2.6

        self.z11 = self.x @ self.Wnx.T # 3.1
        self.z12 = self.z11 + self.bnx # 3.2
        self.z13 = self.hidden @ self.Wnh.T # 3.3
        self.z14 = self.z13 + self.bnh # 3.4
        self.z15 = self.r * self.z14 # 3.5
        self.z16 = self.z12 + self.z15 # 3.6
        self.n = self.n_act.forward(self.z16) # 3.7

        self.z17 = 1 - self.z # 4.1
        self.z18 = self.z17 * self.n # 4.2
        self.z19 = self.z * self.hidden # 4.3
        h_t = self.z18 + self.z19 # 4.4
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)

        return h_t

    def backward(self, delta):

        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        dx = np.zeros_like(self.x)
        dz = np.zeros_like(self.hidden)
        dn = np.zeros_like(self.hidden)
        dr = np.zeros_like(self.hidden)
        dhidden = np.zeros_like(self.hidden)

        dht = delta.flatten()

        # 4.4
        dz18 = dht
        dz19 = dht

        # 4.3
        dz += dz19 * self.hidden
        dhidden += dz19 * self.z

        # 4.2
        dz17 = dz18 * self.n
        dn += dz18 * self.z17

        # 4.1
        dz += - dz17

        # 3.7
        dz16 = dn * self.n_act.backward()
        
        # 3.6
        dz12 = dz16
        dz15 = dz16

        # 3.5
        dr += dz15 * self.z14
        dz14 = dz15 * self.r

        # 3.4
        dz13 = dz14
        self.dbnh += dz14

        # 3.3
        dhidden += dz13 @ self.Wnh
        self.dWnh += (self.hidden.reshape(-1, 1) @ dz13.reshape(1, -1)).T

        # 3.2
        dz11 = dz12
        self.dbnx += dz12

        # 3.1
        dx += dz11 @ self.Wnx
        self.dWnx += (self.x.reshape(-1, 1) @ dz11.reshape(1, -1)).T

        # 2.6
        dz10 = dz * self.z_act.backward()

        # 2.5
        dz7 = dz10
        dz9 = dz10

        # 2.4
        dz8 = dz9
        self.dbzh += dz9

        # 2.3
        dhidden += dz8 @ self.Wzh
        self.dWzh += (self.hidden.reshape(-1, 1) @ dz8.reshape(1, -1)).T

        # 2.2
        dz6 = dz7
        self.dbzx += dz7

        # 2.1
        dx += dz6 @ self.Wzx
        self.dWzx += (self.x.reshape(-1, 1) @ dz6.reshape(1, -1)).T

        # 1.6
        dz5 = dr * self.r_act.backward()

        # 1.5
        dz2 = dz5
        dz4 = dz5

        # 1.4
        dz3 = dz4
        self.dbrh += dz4

        # 1.3
        dhidden += dz3 @ self.Wrh
        self.dWrh += (self.hidden.reshape(-1, 1) @ dz3.reshape(1, -1)).T

        # 1.2
        dz1 = dz2
        self.dbrx += dz2

        # 1.1
        dx += dz1 @ self.Wrx
        self.dWrx += (self.x.reshape(-1, 1) @ dz1.reshape(1, -1)).T

        dx = dx.reshape(1, -1)
        dh_prev_t = dhidden.reshape(1, -1)

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t