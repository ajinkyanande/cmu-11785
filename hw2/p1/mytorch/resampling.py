import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # shapes
        self.A_shape = A.shape
        self.Z_shape = (A.shape[0],
                        A.shape[1],
                        (A.shape[2]*self.upsampling_factor)-(self.upsampling_factor-1))

        # upsample
        Z = np.zeros(self.Z_shape)
        Z[:, :, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        # downsample
        dLdA = dLdZ[:, :, ::self.upsampling_factor]

        # verify shapes
        assert dLdA.shape == self.A_shape
        assert dLdZ.shape == self.Z_shape

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # downsample
        Z = A[:, :, ::self.downsampling_factor]

        # shapes
        self.A_shape = A.shape
        self.Z_shape = Z.shape

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        # upsample
        dLdA = np.zeros(self.A_shape)
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        # verify shapes
        assert dLdZ.shape == self.Z_shape

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # shapes
        self.A_shape = A.shape
        self.Z_shape = (A.shape[0],
                        A.shape[1],
                        (A.shape[2]*self.upsampling_factor)-(self.upsampling_factor-1),
                        (A.shape[3]*self.upsampling_factor)-(self.upsampling_factor-1))

        # upsample
        Z = np.zeros(self.Z_shape)
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # downsample
        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        # verify shapes
        assert dLdA.shape == self.A_shape
        assert dLdZ.shape == self.Z_shape

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        # downsample
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        # shapes
        self.A_shape = A.shape
        self.Z_shape = Z.shape

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        # upsample
        dLdA = np.zeros(self.A_shape)
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        # verify shapes
        assert dLdZ.shape == self.Z_shape

        return dLdA