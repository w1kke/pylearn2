"""
A dataset to convert raw timeseries data to be used in a transformer dataset.
"""
__authors__ = "Robin Lehmann"
__copyright__ = "Copyright 2015, IKI HS-Weingarten-Ravensburg"
__credits__ = ["Robin Lehmann"]
__license__ = "3-clause BSD"
__maintainer__ = "IKI HS-Weingarten-Ravensburg"
__email__ = "lehmannr@hs-weingarten.de"

from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.datasets.timeseries import Timeseries
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import wraps
import numpy as np

class TimeseriesTransformerDataset(TransformerDataset):
    """
    A dataset that applies a transformation on the fly
    as examples are requested.
    """

    def __init__(self, raw, transformer, cpu_only=False,
                 space_preserving=False, block_length=1):
        """
            .. todo::

                WRITEME properly

            Parameters
            ----------
            raw : pylearn2 Dataset
                Provides raw data
            transformer: pylearn2 Block
                To transform the data
            block_length: timeseries length
                Amount of elements of the timeseries
        """
        assert block_length >= 1
      
        if block_length != 1:
            timeseries = Timeseries(X=raw, block_length=block_length)
            super(TimeseriesTransformerDataset,self).__init__(timeseries, transformer, cpu_only, space_preserving)                                           
        else:
            raw = DenseDesignMatrix(X=raw)
            super(TimeseriesTransformerDataset,self).__init__(raw, transformer, cpu_only, space_preserving)

    @wraps(TransformerDataset.get_num_examples)
    def get_num_examples(self):
        if type(self.raw) is Timeseries:
            return self.raw.shape[0]
        else:
            return self.raw.get_num_examples()