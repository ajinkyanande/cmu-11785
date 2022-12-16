import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        self.A = A

        # define shapes
        self.batch_size  = self.A.shape[0]
        self.input_size  = self.A.shape[2]
        self.output_size = 1 + (self.input_size - self.kernel_size) // 1

        # define Z
        self.Z_shape = (self.batch_size, self.out_channels, self.output_size)
        Z = np.zeros(self.Z_shape)

        # Z
        for i in range(0, self.input_size-self.kernel_size+1, 1):
            Z[:, :, i] = np.tensordot(A[:, :, i:i+self.kernel_size], self.W, axes=((1, 2), (1, 2)))

        return Z + self.b[np.newaxis, :, np.newaxis]

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2))
        
        # dLdW
        for i in range(0, self.input_size-self.output_size+1, 1):
            self.dLdW[:, :, i] = np.tensordot(dLdZ, self.A[:, :, i:i+self.output_size], axes=((0,2), (0,2)))

        # flip rows of each filter
        flipped_W = np.flip(self.W, axis=2)
        
        # pad dLdZ
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)))
        output_size_padded = dLdZ_padded.shape[2]

        # define dLdA
        dLdA = np.zeros(self.A.shape)

        # dLdA
        for i in range(0, output_size_padded-self.kernel_size+1, 1):
            dLdA[:, :, i] = np.tensordot(dLdZ_padded[:, :, i:i+self.kernel_size], flipped_W, axes=((1, 2), (0, 2)))

        # verify shapes
        assert dLdA.shape == self.A.shape
        assert dLdZ.shape == self.Z_shape

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):

        self.stride = stride

        # conv wih stride and then downsample
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, weight_init_fn=weight_init_fn,
                                             bias_init_fn=bias_init_fn)

        self.downsample1d = Downsample1d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # conv stride 1 forward
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # downsample backward
        dLdA = self.downsample1d.backward(dLdZ)

        # conv stride 1 backward
        dLdA = self.conv1d_stride1.backward(dLdA)

        return dLdA

class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        # define shapes with stride 1
        self.batch_size  = A.shape[0]
        self.input_rows  = A.shape[2]
        self.input_cols  = A.shape[3]
        self.output_rows = 1 + (self.input_rows - self.kernel_size) // 1
        self.output_cols = 1 + (self.input_cols - self.kernel_size) // 1

        # define Z
        self.Z_shape = (self.batch_size, self.out_channels, self.output_rows, self.output_cols)
        Z = np.zeros(self.Z_shape)

        # Z
        for ri in range(0, self.input_rows-self.kernel_size+1, 1):
            for ci in range(0, self.input_cols-self.kernel_size+1, 1):
                Z[:, :, ri, ci] = np.tensordot(A[:, :, ri:ri+self.kernel_size, ci:ci+self.kernel_size],
                                               self.W,
                                               axes=((1, 2, 3), (1, 2, 3)))

        return Z + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # dLdW
        for ri in range(0, self.input_rows-self.output_rows+1, 1):
            for ci in range(0, self.input_cols-self.output_cols+1, 1):
                self.dLdW[:, :, ri, ci] = np.tensordot(dLdZ,
                                                       self.A[:, :, ri:ri+self.output_rows, ci:ci+self.output_cols],
                                                       axes=((0, 2, 3), (0, 2, 3)))

        # flip rows of each filter
        flipped_W = np.flip(self.W, axis=2)
        flipped_W = np.flip(flipped_W, axis=3)
        
        # pad dLdZ 
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)))
        output_size_padded_rows = dLdZ_padded.shape[2]
        output_size_padded_cols = dLdZ_padded.shape[3]

        # define dLdA
        dLdA = np.zeros(self.A.shape)

        # dLdA
        for ri in range(0, output_size_padded_rows-self.kernel_size+1, 1):
            for ci in range(0, output_size_padded_cols-self.kernel_size+1, 1):
                dLdA[:, :, ri, ci] = np.tensordot(dLdZ_padded[:, :, ri:ri+self.kernel_size, ci:ci+self.kernel_size],
                                                  flipped_W,
                                                  axes=((1, 2, 3), (0, 2, 3)))

        # verify shapes
        assert dLdA.shape == self.A.shape
        assert dLdZ.shape == self.Z_shape

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):

        self.stride = stride

        # conv wih stride and then downsample
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, weight_init_fn=weight_init_fn,
                                             bias_init_fn=bias_init_fn)

        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # conv stride 1 forward
        Z = self.conv2d_stride1.forward(A)

        # downsample forward
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # downsample backward
        dLdA = self.downsample2d.backward(dLdZ)

        # conv stride 1 backward
        dLdA = self.conv2d_stride1.backward(dLdA)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):

        self.upsampling_factor = upsampling_factor

        # upsample then conv
        self.upsample1d = Upsample1d(upsampling_factor=upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, weight_init_fn=weight_init_fn,
                                             bias_init_fn=bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # upsample forward
        A_upsampled = self.upsample1d.forward(A)

        # conv stride 1 forward
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # conv stride 1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)
        
        # upsample backward
        dLdA =  self.upsample1d.backward(dLdA)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):

        self.upsampling_factor = upsampling_factor

        # upsample then conv
        self.upsample2d = Upsample2d(upsampling_factor=upsampling_factor)
        self.conv2d_stride1 = Conv2d_stride1(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, weight_init_fn=weight_init_fn,
                                             bias_init_fn=bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        # upsample forward
        A_upsampled = self.upsample2d.forward(A)

        # conv stride 1 forward
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # conv stride 1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)
        
        # upsample backward
        dLdA =  self.upsample2d.backward(dLdA)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        # define shapes
        self.batch_size = A.shape[0]
        self.in_channels = A.shape[1]
        self.in_width = A.shape[2]

        # flattend each batch individually
        Z = A.reshape(self.batch_size, -1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        # reshape each batch individually
        dLdA = dLdZ.reshape(self.batch_size, self.in_channels, self.in_width)

        return dLdA