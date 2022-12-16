import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.A_shape = A.shape

        # define shapes with stride 1
        self.batch_size = A.shape[0]
        self.channels   = A.shape[1]
        A_input_rows    = A.shape[2]
        A_input_cols    = A.shape[3]
        Z_output_rows   = 1 + (A_input_rows - self.kernel) // 1
        Z_output_cols   = 1 + (A_input_cols - self.kernel) // 1

        # define Z
        self.Z_shape = (self.batch_size, self.channels, Z_output_rows, Z_output_cols)
        Z = np.zeros(self.Z_shape)

        # define storage for max locations in A for each batch and channel
        self.A_max_row = np.zeros(self.Z_shape, dtype=int)
        self.A_max_col = np.zeros(self.Z_shape, dtype=int)

        # Z and location in A
        for bi in range(self.batch_size):
            for chi in range(self.channels):
                for ri in range(0, A_input_rows-self.kernel+1, 1):
                    for ci in range(0, A_input_cols-self.kernel+1, 1):
                        # pick window of wach batch and each channel
                        A_window = A[bi, chi, ri:ri+self.kernel, ci:ci+self.kernel]
                        
                        # flatten window and find max location and unravel to orignal shape
                        A_max_location = np.unravel_index(np.argmax(A_window.flatten()), A_window.shape)

                        # find max
                        Z[bi, chi, ri, ci] += A_window[A_max_location]

                        # save location for backward pass relative to window
                        self.A_max_row[bi, chi, ri, ci], self.A_max_col[bi, chi, ri, ci] = A_max_location
                
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # shapes
        Z_input_rows = dLdZ.shape[2]
        Z_input_cols = dLdZ.shape[3]

        # defne dLdA
        dLdA = np.zeros(self.A_shape)

        for bi in range(self.batch_size):
            for chi in range(self.channels):
                for ri in range(0, Z_input_rows, 1):
                    for ci in range(0, Z_input_cols, 1):
                        A_max_row = self.A_max_row[bi, chi, ri, ci]
                        A_max_col = self.A_max_col[bi, chi, ri, ci]
                        dLdA[bi, chi, ri+A_max_row, ci+A_max_col] += dLdZ[bi, chi, ri, ci]

        # check shapes
        assert dLdZ.shape == self.Z_shape
        assert dLdA.shape == self.A_shape

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.A_shape = A.shape

        # define shapes with stride 1
        self.batch_size = A.shape[0]
        self.channels   = A.shape[1]
        A_input_rows    = A.shape[2]
        A_input_cols    = A.shape[3]
        Z_output_rows   = 1 + (A_input_rows - self.kernel) // 1
        Z_output_cols   = 1 + (A_input_cols - self.kernel) // 1

        # define Z
        self.Z_shape = (self.batch_size, self.channels, Z_output_rows, Z_output_cols)
        Z = np.zeros(self.Z_shape)

        # Z
        for bi in range(self.batch_size):
            for chi in range(self.channels):
                for ri in range(0, A_input_rows-self.kernel+1, 1):
                    for ci in range(0, A_input_cols-self.kernel+1, 1):
                        # pick window of wach batch and each channel
                        A_window = A[bi, chi, ri:ri+self.kernel, ci:ci+self.kernel]

                        # find mean
                        Z[bi, chi, ri, ci] += np.mean(A_window)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # shapes
        Z_input_rows = dLdZ.shape[2]
        Z_input_cols = dLdZ.shape[3]

        # defne dLdA
        dLdA = np.zeros(self.A_shape)

        for bi in range(self.batch_size):
            for chi in range(self.channels):
                for ri in range(0, Z_input_rows, 1):
                    for ci in range(0, Z_input_cols, 1):
                        dLdA[bi, chi, ri:ri+self.kernel, ci:ci+self.kernel] += dLdZ[bi, chi, ri, ci] / (self.kernel ** 2)

        # check shapes
        assert dLdZ.shape == self.Z_shape
        assert dLdA.shape == self.A_shape

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel=self.kernel)
        self.downsample2d      = Downsample2d(downsampling_factor=self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        # maxpool stride 1 forward
        Z = self.maxpool2d_stride1.forward(A)

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

        # maxpool stride 1 backward
        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel=self.kernel)
        self.downsample2d = Downsample2d(downsampling_factor=self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # meanpool stride 1 forward
        Z = self.meanpool2d_stride1.forward(A)

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

        # meanpool stride 1 backward
        dLdA = self.meanpool2d_stride1.backward(dLdA)

        return dLdA
