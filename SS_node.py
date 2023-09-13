from skmultiflow.bayes import do_naive_bayes_prediction
import numpy as np
from skmultiflow.utils import get_dimensions
from skmultiflow.trees.gaussian_estimator import GaussianEstimator
from SS_attribute_observer import AttributeSplitSuggestion
from SS_attribute_observer import AttributeClassObserverNull
from SS_attribute_observer import NominalAttributeClassObserver
from SS_attribute_observer import NumericAttributeClassObserverGaussian

import textwrap
from abc import ABCMeta

import copy as cp

class Node(metaclass=ABCMeta):
    """ Base class for nodes in a Hoeffding Tree.

    Parameters
    ----------
    class_observations: dict (class_value, weight) or None
        Class observations.
    """

    def __init__(self, class_observations=None):
        self._ss_observers = 0.0
        """ Node class constructor. """
        if class_observations is None:
            class_observations = {}  # Dictionary (class_value, weight)
        self._observed_class_distribution = class_observations

    @staticmethod
    def is_leaf():
        """ Determine if the node is a leaf.

        Returns
        -------
        True if leaf, False otherwise

        """
        return True

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
           Data instances.
        parent: skmultiflow.trees.nodes.Node or None
            Parent node.
        parent_branch: Int
            Parent branch index

        Returns
        -------
        FoundNode
            The corresponding leaf.

        """
        return FoundNode(self, parent, parent_branch)

    def get_observed_class_distribution(self):
        """ Get the current observed class distribution at the node.

        Returns
        -------
        dict (class_value, weight)
            Class distribution at the node.

        """
        return self._observed_class_distribution

    def set_observed_class_distribution(self, observed_class_distribution):
        """ Set the observed class distribution at the node.

        Parameters
        -------
        dict (class_value, weight)
            Class distribution at the node.

        """
        self._observed_class_distribution = observed_class_distribution

    def get_class_votes(self, X, ht):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
           Data instances.
        ht: HoeffdingTreeClassifier
            The Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """

        # class_size = ht.return_w()
        #
        # ss_observed_class_distribution = cp.deepcopy(self._observed_class_distribution)
        #
        # for i in range(len(class_size)):
        #     weight = 1 / (class_size[i] + 1e-8)
        #     if weight > 1e+5:
        #         weight = 1e+5
        #     if i in ss_observed_class_distribution:
        #         ss_observed_class_distribution[i] = ss_observed_class_distribution[i] * weight

        return self._observed_class_distribution

    def observed_class_distribution_is_pure(self):
        """ Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
        boolean
            True if observed number of classes is less than 2, False otherwise.

        """
        count = 0
        for _, weight in self._observed_class_distribution.items():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2

    def subtree_depth(self):
        """ Calculate the depth of the subtree from this node.

        Returns
        -------
        int
            Subtree depth, 0 if the node is a leaf.

        """
        return 0

    def calculate_promise(self):
        """ Calculate node's promise.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self._observed_class_distribution.values())
        if total_seen > 0:
            return total_seen - max(self._observed_class_distribution.values())
        else:
            return 0

    def describe_subtree(self, ht, buffer, indent=0):
        """ Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        ht: HoeffdingTreeClassifier
            The tree to describe.
        buffer: string
            The string buffer where the tree's structure will be stored
        indent: int
            Indentation level (number of white spaces for current node.)

        """
        buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)

        if ht._estimator_type == 'classifier':
            class_val = max(
                self._observed_class_distribution,
                key=self._observed_class_distribution.get
            )
            buffer[0] += 'Class {} | {}\n'.format(
                class_val, self._observed_class_distribution
            )
        else:
            text = '{'
            for i, (k, v) in enumerate(self._observed_class_distribution.items()):
                # Multi-target regression case
                if hasattr(v, 'shape') and len(v.shape) > 0:
                    text += '{}: ['.format(k)
                    text += ', '.join(['{:.4f}'.format(e) for e in v.tolist()])
                    text += ']'
                else:  # Single-target regression
                    text += '{}: {:.4f}'.format(k, v)
                text += ', ' if i < len(self._observed_class_distribution) - 1 else ''
            text += '}'
            buffer[0] += 'Statistics {}\n'.format(text)  # Regression problems

    # TODO
    def get_description(self):
        pass



class LearningNode(Node):
    """ Base class for Learning Nodes in a Hoeffding Tree.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations=None):
        """ LearningNode class constructor. """
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree to update.

        """
        pass



class ActiveLearningNode(LearningNode):
    """ Learning node that supports growth.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ ActiveLearningNode class constructor. """
        super().__init__(initial_class_observations)
        self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
        self._attribute_observers = {}
        self._ss_attribute_observers = {}

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree to update.
        """
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
            self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))

        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], int(y), weight)

    # def get_weight_seen(self):
    #     """ Calculate the total weight seen by the node.
    #
    #     Returns
    #     -------
    #     float
    #         Total weight seen.
    #
    #     """
    #     return sum(self._observed_class_distribution.values())


    def get_weight_seen(self):
        """
        未标记样本同样可以促进树的生长
        """
        # weight = sum(self._observed_class_distribution.values())
        # weight = weight + self._ss_observers
        weight = self._ss_observers

        return weight

    def get_weight_seen_at_last_split_evaluation(self):
        """ Retrieve the weight seen at last split evaluation.

        Returns
        -------
        float
            Weight seen at last split evaluation.

        """
        return self._weight_seen_at_last_split_evaluation

    def set_weight_seen_at_last_split_evaluation(self, weight):
        """ Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight: float
            Weight seen at last split evaluation.

        """
        self._weight_seen_at_last_split_evaluation = weight

    # def get_best_split_suggestions(self, criterion, ht):
    #     """ Find possible split candidates.
    #
    #     Parameters
    #     ----------
    #     criterion: SplitCriterion
    #         The splitting criterion to be used.
    #     ht: HoeffdingTreeClassifier
    #         Hoeffding Tree.
    #
    #     Returns
    #     -------
    #     list
    #         Split candidates.
    #
    #     """
    #     class_size = ht.return_w()
    #     best_suggestions = []
    #     pre_split_dist = self._observed_class_distribution
    #     if not ht.no_preprune:
    #         # Add null split as an option
    #         null_split = AttributeSplitSuggestion(
    #             None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
    #         )
    #         best_suggestions.append(null_split)
    #     for i, obs in self._attribute_observers.items():
    #         best_suggestion = obs.get_best_evaluated_split_suggestion(
    #             criterion, pre_split_dist, i, ht.binary_split, class_size
    #         )
    #         if best_suggestion is not None:
    #             best_suggestions.append(best_suggestion)
    #     return best_suggestions


    def ss_learn_from_instance(self, X, weight, ht):
        """
        从未标记样本中选定的每个特征下学习一个高斯分布

        每个高斯分布有对应的权重，代表学习到的样本总数
        """
        self._ss_observers = self._ss_observers + weight
        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])
        for i in self.list_attributes:
            try:
                obs = self._ss_attribute_observers[i]
            except KeyError:
                obs = GaussianEstimator()
                self._ss_attribute_observers[i] = obs
            obs.add_observation(X[i], weight)

    def get_best_split_suggestions(self, criterion, ht):
        """ 覆写了主函数的分裂函数
        """
        class_size = ht.return_w()
        best_suggestions = []
        pre_split_dist = self._observed_class_distribution
        if not ht.no_preprune:
            # Add null split as an option
            null_split = AttributeSplitSuggestion(
                None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        if self._ss_attribute_observers == {}:
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(
                    criterion, pre_split_dist, i, ht.binary_split, class_size
                )
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
        else:
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.ss_get_best_evaluated_split_suggestion(
                    criterion, pre_split_dist, i, ht.binary_split, class_size, self._ss_attribute_observers[i]
                )
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
        return best_suggestions

    def disable_attribute(self, att_idx):
        """ Disable an attribute observer.

        Parameters
        ----------
        att_idx: int
            Attribute index.

        """
        if att_idx in self._attribute_observers:
            self._attribute_observers[att_idx] = AttributeClassObserverNull()

    def get_attribute_observers(self):
        """ Get attribute observers at this node.

        Returns
        -------
        dict (attribute id, attribute observer object)
            Attribute observers of this node.

        """
        return self._attribute_observers

    def set_attribute_observers(self, attribute_observers):
        """ set attribute observers.

        Parameters
        ----------
        attribute_observers: dict (attribute id, attribute observer object)
            new attribute observers.

        """
        self._attribute_observers = attribute_observers



class LearningNodeNB(ActiveLearningNode):
    """ Learning node that uses Naive Bayes models.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_class_observations)

    def get_class_votes(self, X, ht):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """

        class_size = ht.return_w()

        ss_observed_class_distribution = cp.deepcopy(self._observed_class_distribution)

        for i in range(len(class_size)):
            weight = 1 / (class_size[i] + 1e-8)
            if weight > 1e+5:
                weight = 1e+5
            if i in ss_observed_class_distribution:
                ss_observed_class_distribution[i] = ss_observed_class_distribution[i] * weight

        if self.get_weight_seen() >= ht.nb_threshold:
            return do_naive_bayes_prediction(
                X, ss_observed_class_distribution, self._attribute_observers
            )
        else:
            return super().get_class_votes(X, ht)

    def disable_attribute(self, att_index):
        """ Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index: int
            Attribute index.

        """
        pass



class LearningNodeNBAdaptive(LearningNodeNB):
    """ Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations):
        """ LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_class_observations)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            The instance's weight.
        ht: HoeffdingTreeClassifier
            The Hoeffding Tree to update.

        """
        if self._observed_class_distribution == {}:
            # All classes equal, default to class 0
            if 0 == y:
                self._mc_correct_weight += weight
        elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
            self._mc_correct_weight += weight
        nb_prediction = do_naive_bayes_prediction(
            X, self._observed_class_distribution, self._attribute_observers
        )
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight
        super().learn_from_instance(X, y, weight, ht)

    def get_class_votes(self, X, ht):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """

        class_size = ht.return_w()

        ss_observed_class_distribution = cp.deepcopy(self._observed_class_distribution)

        for i in range(len(class_size)):
            weight = 1 / (class_size[i] + 1e-8)
            if weight > 1e+5:
                weight = 1e+5
            if i in ss_observed_class_distribution:
                ss_observed_class_distribution[i] = ss_observed_class_distribution[i] * weight

        if self._mc_correct_weight > self._nb_correct_weight:
            return self._observed_class_distribution


        return do_naive_bayes_prediction(
            X, ss_observed_class_distribution, self._attribute_observers
        )


class RandomLearningNodeClassification(ActiveLearningNode):
    """ARF learning node class.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, max_features, random_state=None):
        """ RandomLearningNodeClassification class constructor. """
        super().__init__(initial_class_observations)

        self.max_features = max_features
        self._attribute_observers = {}
        self.list_attributes = np.array([])
        self.random_state = random_state

    def learn_from_instance(self, X, y, weight, ht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree to update.
        """
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
            self._observed_class_distribution = dict(
                sorted(self._observed_class_distribution.items())
            )

        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])

        for i in self.list_attributes:
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], int(y), weight)

    def _sample_features(self, n_features):
        return self.random_state.choice(
            n_features, size=self.max_features, replace=False
        )


class RandomLearningNodeNB(RandomLearningNodeClassification):
    """ARF Naive Bayes learning node class.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_class_observations, max_features, random_state):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_class_observations, max_features, random_state)

    def get_class_votes(self, X, ht):
        """Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """

        class_size = ht.return_w()

        ss_observed_class_distribution = cp.deepcopy(self._observed_class_distribution)

        for i in range(len(class_size)):
            weight = 1 / (class_size[i] + 1e-8)
            if weight > 1e+5:
                weight = 1e+5
            if i in ss_observed_class_distribution:
                ss_observed_class_distribution[i] = ss_observed_class_distribution[i] * weight


        if self.get_weight_seen() >= ht.nb_threshold:
            return do_naive_bayes_prediction(
                X, ss_observed_class_distribution, self._attribute_observers
            )
        else:
            return super().get_class_votes(X, ht)


class RandomLearningNodeNBAdaptive(RandomLearningNodeNB):
    """Naive Bayes Adaptive learning node class.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, max_features, random_state):
        """LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_class_observations, max_features, random_state)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0
        self._ss_attribute_observers = {}


    def get_weight_seen(self):
        """
        未标记样本同样可以促进树的生长
        """
        # weight = sum(self._observed_class_distribution.values())
        # weight = weight + self._ss_observers
        weight = self._ss_observers

        return weight

    def ss_learn_from_instance(self, X, weight, ht):
        """
        从未标记样本中选定的每个特征下学习一个高斯分布

        每个高斯分布有对应的权重，代表学习到的样本总数
        """
        self._ss_observers = self._ss_observers + weight
        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])
        for i in self.list_attributes:
            try:
                obs = self._ss_attribute_observers[i]
            except KeyError:
                obs = GaussianEstimator()
                self._ss_attribute_observers[i] = obs
            obs.add_observation(X[i], weight)

    def get_best_split_suggestions(self, criterion, ht):
        """ 覆写了主函数的分裂函数
        """
        class_size = ht.return_w()
        best_suggestions = []
        pre_split_dist = self._observed_class_distribution
        if not ht.no_preprune:
            # Add null split as an option
            null_split = AttributeSplitSuggestion(
                None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        if self._ss_attribute_observers == {}:
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(
                    criterion, pre_split_dist, i, ht.binary_split, class_size
                )
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
        else:
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.ss_get_best_evaluated_split_suggestion(
                    criterion, pre_split_dist, i, ht.binary_split, class_size, self._ss_attribute_observers[i]
                )
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
        return best_suggestions

    def learn_from_instance(self, X, y, weight, ht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            The instance's weight.
        ht: HoeffdingTreeClassifier
            The Hoeffding Tree to update.
        """
        if self._observed_class_distribution == {}:
            # All classes equal, default to class 0
            if 0 == y:
                self._mc_correct_weight += weight
        elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
            self._mc_correct_weight += weight
        nb_prediction = do_naive_bayes_prediction(
            X, self._observed_class_distribution, self._attribute_observers
        )
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight
        super().learn_from_instance(X, y, weight, ht)

    def get_class_votes(self, X, ht):
        """进行了权重修正，确保预测不受类别规模的影响
        """
        if self._mc_correct_weight > self._nb_correct_weight:
            return self._observed_class_distribution
        class_size = ht.return_w()

        ss_observed_class_distribution = cp.deepcopy(self._observed_class_distribution)

        for i in range(len(class_size)):
            weight = 1 / (class_size[i]+1e-8)
            if weight > 1e+5:
                weight = 1e+5
            if i in ss_observed_class_distribution:
                ss_observed_class_distribution[i] = ss_observed_class_distribution[i] * weight

        return do_naive_bayes_prediction(
            X, ss_observed_class_distribution, self._attribute_observers
        )

class SplitNode(Node):
    """ Node that splits the data in a Hoeffding Tree.

    Parameters
    ----------
    split_test: InstanceConditionalTest
        Split test.
    class_observations: dict (class_value, weight) or None
        Class observations

    """

    def __init__(self, split_test, class_observations):
        """ SplitNode class constructor."""
        super().__init__(class_observations)
        self._split_test = split_test
        # Dict of tuples (branch, child)
        self._children = {}

    def num_children(self):
        """ Count the number of children for a node."""
        return len(self._children)

    def get_split_test(self):
        """ Retrieve the split test of this node.

        Returns
        -------
        InstanceConditionalTest
            Split test.

        """

        return self._split_test

    def set_child(self, index, node):
        """ Set node as child.

        Parameters
        ----------
        index: int
            Branch index where the node will be inserted.

        node: skmultiflow.trees.nodes.Node
            The node to insert.

        """
        if (self._split_test.max_branches() >= 0) and (index >= self._split_test.max_branches()):
            raise IndexError
        self._children[index] = node

    def get_child(self, index):
        """ Retrieve a node's child given its branch index.

        Parameters
        ----------
        index: int
            Node's branch index.

        Returns
        -------
        skmultiflow.trees.nodes.Node or None
            Child node.

        """
        if index in self._children:
            return self._children[index]
        else:
            return None

    def instance_child_index(self, X):
        """ Get the branch index for a given instance at the current node.

        Returns
        -------
        int
            Branch index, -1 if unknown.

        """
        return self._split_test.branch_for_instance(X)

    @staticmethod
    def is_leaf():
        """ Determine if the node is a leaf.

        Returns
        -------
        boolean
            True if node is a leaf, False otherwise

        """
        return False

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
           Data instances.
        parent: skmultiflow.trees.nodes.Node
            Parent node.
        parent_branch: int
            Parent branch index.

        Returns
        -------
        FoundNode
            Leaf node for the instance.

        """
        child_index = self.instance_child_index(X)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                return child.filter_instance_to_leaf(X, self, child_index)
            else:
                return FoundNode(None, self, child_index)
        else:
            return FoundNode(self, parent, parent_branch)

    def subtree_depth(self):
        """ Calculate the depth of the subtree from this node.

        Returns
        -------
        int
            Subtree depth, 0 if node is a leaf.
        """
        max_child_depth = 0
        for child in self._children.values():
            if child is not None:
                depth = child.subtree_depth()
                if depth > max_child_depth:
                    max_child_depth = depth
        return max_child_depth + 1

    def describe_subtree(self, ht, buffer, indent=0):
        """ Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        ht: HoeffdingTreeClassifier
            The tree to describe.
        buffer: string
            The buffer where the tree's structure will be stored.
        indent: int
            Indentation level (number of white spaces for current node).

        """
        for branch_idx in range(self.num_children()):
            child = self.get_child(branch_idx)
            if child is not None:
                buffer[0] += textwrap.indent('if ', ' ' * indent)
                buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                buffer[0] += ':\n'
                child.describe_subtree(ht, buffer, indent + 2)

    def get_predicate(self, branch):

        return self._split_test.branch_rule(branch)

class FoundNode(object):
    """ Base class for tree nodes.

    Parameters
    ----------
    node: SplitNode or LearningNode
        The node object.
    parent: SplitNode or None
        The node's parent.
    parent_branch: int
        The parent node's branch.
    depth: int
        Depth of the tree where the node is located.

    """

    def __init__(self, node=None, parent=None, parent_branch=None, depth=None):
        """ FoundNode class constructor. """
        self.node = node
        self.parent = parent
        self.parent_branch = parent_branch
        self.depth = depth

class InactiveLearningNode(LearningNode):
    """ Inactive learning node that does not grow.

    Parameters
    ----------
    initial_class_observations: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_class_observations=None):
        """ InactiveLearningNode class constructor. """
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree to update.

        """
        try:
            self._observed_class_distribution[y] += weight
        except KeyError:
            self._observed_class_distribution[y] = weight
            self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))


