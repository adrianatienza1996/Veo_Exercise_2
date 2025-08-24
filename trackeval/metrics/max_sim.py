import numpy as np
from ._base_metric import _BaseMetric
from .. import _timing


class MaxSim(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self, config=None):
        super().__init__()
        self.fields = ['Max_Sim']
        self.float_fields = self.fields
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        """Returns Maximum Similarity for any pair in the sequence""" 
        sim = data['similarity_scores']
        res_max = -1
        
        for t in sim:
            try:
                res_max = max(res_max, np.max(t))
            
            except:
                continue

        res = {'Max_Sim': res_max}
        
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.float_fields:
            res[field] = self._combine_sum(all_res, field)

        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=None):
        """Combines metrics across all classes by averaging over the class values"""
        res = {}
        for field in self.float_fields:
            res[field] = self._combine_sum(all_res, field)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.float_fields:
            res[field] = self._combine_sum(all_res, field)
        return res
