"""
Implementation of a time series class based on the DenseDesignMatrix class.
It is based on the assumption that the observations in the time series all have 
the same amount of values. These are stored in the DenseDesignMatrix.
During the use of the dataset the iterator will use a fixed amount of these 
observations.
"""
__authors__ = "Robin Lehmann"
__copyright__ = "Copyright 2015, IKI HS-Weingarten-Ravensburg"
__credits__ = ["Robin Lehmann"]
__license__ = "3-clause BSD"
__maintainer__ = "IKI HS-Weingarten-Ravensburg"
__email__ = "lehmannr@hs-weingarten.de"

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from theano import shared
import numpy as np

class Timeseries(DenseDesignMatrix):

    _default_seed = (17, 2, 946)

    def __init__(self, X=None, topo_view=None, y=None,
                view_converter=None, axes=('b', 0, 1, 'c'),
                rng=_default_seed, preprocessor=None, fit_preprocessor=False,
                X_labels=None, y_labels=None, block_length=1):

        assert block_length >= 1
        
        if block_length != 1:
            if y_labels == None:
                timeseries = np.reshape(X[0:(X.shape[0] - X.shape[0] % block_length)], 
                        (X[0:(X.shape[0] - X.shape[0] % block_length)].shape[0]/block_length, -1))
            else:
                timeseries = np.reshape(X[0:(X.shape[0] - X.shape[0] % block_length), range(len(X[0]) - 1)],
                                        (X[0:(X.shape[0] - X.shape[0] % block_length)].shape[0]/block_length, -1))
                y = np.reshape(X[0:(X.shape[0] - X.shape[0] % block_length), -1],
                                        (X[0:(X.shape[0] - X.shape[0] % block_length)].shape[0]/block_length, -1))
                y = y[:,0].astype(int)

            #view_converter = DefaultViewConverter((1, timeseries.shape[1], 1))

            super(Timeseries,self).__init__(timeseries, topo_view, y, view_converter, 
                                    axes, rng, preprocessor, fit_preprocessor, X_labels, y_labels)
            self.shape = timeseries.shape
        else:
            view_converter = DefaultViewConverter((1, X.shape[1], 1))

            super(Timeseries,self).__init__(X, topo_view, y, view_converter, axes, 
                                    rng, preprocessor, fit_preprocessor, X_labels, y_labels)
            self.shape = X.shape
