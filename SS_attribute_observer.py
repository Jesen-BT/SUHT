from abc import ABCMeta, abstractmethod
import numpy as np
from skmultiflow.trees.gaussian_estimator import GaussianEstimator
from skmultiflow.trees.attribute_test import NumericAttributeBinaryTest
from sortedcontainers.sortedlist import SortedList

from skmultiflow.trees.attribute_test import NominalAttributeBinaryTest
from skmultiflow.trees.attribute_test import NominalAttributeMultiwayTest

class AttributeClassObserver(metaclass=ABCMeta):
    """Abstract class for observing the class data distribution for an attribute.
    This observer monitors the class distribution of a given attribute.

    This class should not be instantiated, as none of its methods are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def observe_attribute_class(self, att_val, class_val, weight):
        """Update statistics of this observer given an attribute value, a class
        and the weight of the instance observed.

        Parameters
        ----------
        att_val : float
            The value of the attribute.

        class_val: int
            The class value.

        weight: float
            The weight of the instance.

        """
        raise NotImplementedError

    @abstractmethod
    def probability_of_attribute_value_given_class(self, att_val, class_val):
        """Get the probability for an attribute value given a class.

        Parameters
        ----------
        att_val: float
            The value of the attribute.

        class_val: int
            The class value.

        Returns
        -------
        float
            Probability for an attribute value given a class.

        """
        raise NotImplementedError

    @abstractmethod
    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only, class_size):
        """ get_best_evaluated_split_suggestion

        Gets the best split suggestion given a criterion and a class distribution

        Parameters
        ----------
        criterion: The split criterion to use
        pre_split_dist: The class distribution before the split
        att_idx: The attribute index
        binary_only: True to use binary splits

        Returns
        -------
        Suggestion of best attribute split

        """
        raise NotImplementedError

class AttributeSplitSuggestion(object):
    def __init__(self, split_test, resulting_class_distributions, merit):
        self.split_test = split_test
        self.resulting_class_distributions = resulting_class_distributions
        self.merit = merit

    def num_splits(self):
        return len(self.resulting_class_distributions)

    def resulting_class_distribution_from_split(self, split_idx):
        return self.resulting_class_distributions[split_idx]

class AttributeClassObserverNull(AttributeClassObserver):
    """ Class for observing the class data distribution for a null attribute.
    This method is used to disable the observation for an attribute.
    Used in decision trees to monitor data statistics on leaves.

    """
    def __init__(self):
        super().__init__()

    def observe_attribute_class(self, att_val, class_val, weight):
        pass

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        return None

class NumericAttributeClassObserverGaussian(AttributeClassObserver):
    """ Class for observing the class data distribution for a numeric attribute
    using gaussian estimators.
    """

    def __init__(self):
        super().__init__()
        self._min_value_observed_per_class = {}
        self._max_value_observed_per_class = {}
        self._att_val_dist_per_class = {}
        self.num_bin_options = 10  # The number of bins, default 10

    def observe_attribute_class(self, att_val, class_val, weight):
        if att_val is None:
            return
        else:
            try:
                val_dist = self._att_val_dist_per_class[class_val]
                if att_val < self._min_value_observed_per_class[class_val]:
                    self._min_value_observed_per_class[class_val] = att_val
                if att_val > self._max_value_observed_per_class[class_val]:
                    self._max_value_observed_per_class[class_val] = att_val
            except KeyError:
                val_dist = GaussianEstimator()
                self._att_val_dist_per_class[class_val] = val_dist
                self._min_value_observed_per_class[class_val] = att_val
                self._max_value_observed_per_class[class_val] = att_val
                self._att_val_dist_per_class = dict(sorted(self._att_val_dist_per_class.items()))
                self._max_value_observed_per_class = dict(sorted(self._max_value_observed_per_class.items()))
                self._min_value_observed_per_class = dict(sorted(self._min_value_observed_per_class.items()))

            val_dist.add_observation(att_val, weight)

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        if class_val in self._att_val_dist_per_class:
            obs = self._att_val_dist_per_class[class_val]
            return obs.probability_density(att_val)
        else:
            return 0.0

    def ss_get_best_evaluated_split_suggestion(self,
                                               criterion,
                                               pre_split_dist,
                                               att_idx,
                                               binary_only,
                                               class_size,
                                               ss_attribute_observer,):
        """
        半监督下的节点分裂，使用未标记样本指导分裂结果
        """
        best_suggestion = None
        suggested_split_values = self.get_split_point_suggestions()
        for split_value in suggested_split_values:
            post_split_dist = self.get_class_dists_from_binary_split(split_value)

            l_sum = sum(post_split_dist[0].values())
            r_sum = sum(post_split_dist[1].values())

            for i in range(len(class_size)):
                weight = 1/(class_size[i]+1e-8)
                if weight > 1e+5:
                    weight = 1e+5
                for _ in post_split_dist:
                    if i in _:
                        _[i] = _[i] * weight

            new_l_sum = sum(post_split_dist[0].values()) + 1e-8
            new_r_sum = sum(post_split_dist[1].values()) + 1e-8

            for k, v in post_split_dist[0].items():
                post_split_dist[0][k] = (post_split_dist[0][k] / new_l_sum) * l_sum

            for k, v in post_split_dist[1].items():
                post_split_dist[1][k] = (post_split_dist[1][k] / new_r_sum) * r_sum

            weight = ss_attribute_observer.estimated_weight_lessthan_equalto_greaterthan_value(split_value)
            lweight = (weight[0] + weight[1])/sum(weight)
            rweight = weight[2]/sum(weight)

            sum_w = 0
            for i in range(2):
                for k, v in post_split_dist[i].items():
                    sum_w = sum_w + v

            lweight = lweight * sum_w
            rweight = rweight * sum_w

            l_sum = 0
            for k, v in post_split_dist[0].items():
                l_sum = l_sum + v + 1e-8
            r_sum = 0
            for k, v in post_split_dist[1].items():
                r_sum = r_sum + v + 1e-8

            post_split_dist[0] = {k: (v/l_sum) * lweight for k, v in post_split_dist[0].items()}
            post_split_dist[1] = {k: (v/r_sum) * rweight for k, v in post_split_dist[1].items()}

            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            post_split_dist = self.get_class_dists_from_binary_split(split_value)
            if best_suggestion is None or merit > best_suggestion.merit:
                num_att_binary_test = NumericAttributeBinaryTest(att_idx, split_value, True)
                best_suggestion = AttributeSplitSuggestion(num_att_binary_test, post_split_dist, merit)
        return best_suggestion

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only, class_size):
        best_suggestion = None
        suggested_split_values = self.get_split_point_suggestions()
        for split_value in suggested_split_values:
            post_split_dist = self.get_class_dists_from_binary_split(split_value)
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                num_att_binary_test = NumericAttributeBinaryTest(att_idx, split_value, True)
                best_suggestion = AttributeSplitSuggestion(num_att_binary_test, post_split_dist, merit)
        return best_suggestion

    def get_split_point_suggestions(self):
        suggested_split_values = SortedList()
        min_value = np.inf
        max_value = -np.inf
        for k, estimator in self._att_val_dist_per_class.items():
            if self._min_value_observed_per_class[k] < min_value:
                min_value = self._min_value_observed_per_class[k]
            if self._max_value_observed_per_class[k] > max_value:
                max_value = self._max_value_observed_per_class[k]
        if min_value < np.inf:
            bin_size = max_value - min_value
            bin_size /= (float(self.num_bin_options) + 1.0)
            for i in range(self.num_bin_options):
                split_value = min_value + (bin_size * (i + 1))
                if split_value > min_value and split_value < max_value:
                    suggested_split_values.add(split_value)
        return suggested_split_values

    def get_class_dists_from_binary_split(self, split_value):
        """
        Assumes all values equal to split_value go to lhs
        """
        lhs_dist = {}
        rhs_dist = {}
        for k, estimator in self._att_val_dist_per_class.items():
            if split_value < self._min_value_observed_per_class[k]:
                rhs_dist[k] = estimator.get_total_weight_observed()
            elif split_value >= self._max_value_observed_per_class[k]:
                lhs_dist[k] = estimator.get_total_weight_observed()
            else:
                weight_dist = estimator.estimated_weight_lessthan_equalto_greaterthan_value(split_value)
                lhs_dist[k] = weight_dist[0] + weight_dist[1]
                rhs_dist[k] = weight_dist[2]
        return [lhs_dist, rhs_dist]

class NominalAttributeClassObserver(AttributeClassObserver):
    """ Class for observing the class data distribution for a nominal attribute.
    This observer monitors the class distribution of a given attribute.
    Used in naive Bayes and decision trees to monitor data statistics on leaves.

    """

    def __init__(self):
        super().__init__()
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_val_dist_per_class = {}

    def observe_attribute_class(self, att_val, class_val, weight):
        if att_val is None:
            self._missing_weight_observed += weight
        else:
            try:
                val_dist = self._att_val_dist_per_class[class_val]
            except KeyError:
                self._att_val_dist_per_class[class_val] = {att_val: 0.0}
                self._att_val_dist_per_class = dict(sorted(self._att_val_dist_per_class.items()))
            try:
                self._att_val_dist_per_class[class_val][att_val] += weight
            except KeyError:
                self._att_val_dist_per_class[class_val][att_val] = weight
                self._att_val_dist_per_class[class_val] = dict(
                    sorted(self._att_val_dist_per_class[class_val].items())
                )

        self._total_weight_observed += weight

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        obs = self._att_val_dist_per_class.get(class_val, None)
        if obs is not None:
            value = obs[att_val] if att_val in obs else 0.0
            return (value + 1.0) / (sum(obs.values()) + len(obs))
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        best_suggestion = None
        att_values = sorted(set(
            [att_val for att_val_per_class in self._att_val_dist_per_class.values()
             for att_val in att_val_per_class]
        ))
        if not binary_only:
            post_split_dist = self.get_class_dist_from_multiway_split()
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            branch_mapping = {attr_val: branch_id for branch_id, attr_val in
                              enumerate(att_values)}
            best_suggestion = AttributeSplitSuggestion(
                NominalAttributeMultiwayTest(att_idx, branch_mapping),
                post_split_dist, merit
            )
        for att_val in att_values:
            post_split_dist = self.get_class_dist_from_binary_split(att_val)
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                best_suggestion = AttributeSplitSuggestion(
                    NominalAttributeBinaryTest(att_idx, att_val),
                    post_split_dist, merit
                )
        return best_suggestion

    def get_class_dist_from_multiway_split(self):
        resulting_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j not in resulting_dist:
                    resulting_dist[j] = {}
                if i not in resulting_dist[j]:
                    resulting_dist[j][i] = 0.0
                resulting_dist[j][i] += value

        sorted_keys = sorted(resulting_dist.keys())
        distributions = [
            dict(sorted(resulting_dist[k].items())) for k in sorted_keys
        ]
        return distributions

    def get_class_dist_from_binary_split(self, val_idx):
        equal_dist = {}
        not_equal_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j == val_idx:
                    if i not in equal_dist:
                        equal_dist[i] = 0.0
                    equal_dist[i] += value
                else:
                    if i not in not_equal_dist:
                        not_equal_dist[i] = 0.0
                    not_equal_dist[i] += value
        return [equal_dist, not_equal_dist]



