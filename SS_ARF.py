from copy import deepcopy
import math
import itertools

from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTreeClassifier

import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from SS_HoeffdingTree import ARFHoeffdingTreeClassifier
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from skmultiflow.utils import get_dimensions, normalize_values_in_dict, check_random_state,\
    check_weights


class SemiAdaptiveRandomForestClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self,
                 n_estimators=20,
                 max_features='auto',
                 disable_weighted_vote=False,
                 lambda_value=6,
                 performance_metric='acc',
                 drift_detection_method: BaseDriftDetector=ADWIN(0.001),
                 warning_detection_method: BaseDriftDetector=ADWIN(0.01),
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_criterion='info_gain',
                 split_confidence=0.1,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 random_state=None):
        """AdaptiveRandomForestClassifier class constructor."""
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.disable_weighted_vote = disable_weighted_vote
        self.lambda_value = lambda_value
        if isinstance(drift_detection_method, BaseDriftDetector):
            self.drift_detection_method = drift_detection_method
        else:
            self.drift_detection_method = None
        if isinstance(warning_detection_method, BaseDriftDetector):
            self.warning_detection_method = warning_detection_method
        else:
            self.warning_detection_method = None
        self.instances_seen = 0
        self.classes = None
        self._train_weight_seen_by_model = 0.0
        self.ensemble = None
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)   # Actual random_state object
        if performance_metric in ['acc', 'kappa']:
            self.performance_metric = performance_metric
        else:
            raise ValueError('Invalid performance metric: {}'.format(performance_metric))

        # ARH Hoeffding Tree configuration
        self.max_byte_size = max_byte_size
        self. memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

    def ss_partial_fit(self, X):
        row_cnt, _ = get_dimensions(X)
        for i in range(row_cnt):
            for j in range(self.n_estimators):
                k = self._random_state.poisson(self.lambda_value)
                if k > 0:
                    self.ensemble[j].ss_partial_fit(np.asarray([X[i]]),
                                                 sample_weight=np.asarray([k]))


    def partial_fit(self, X, y, classes=None, sample_weight=None):

        if self.classes is None and classes is not None:
            self.classes = classes

        if sample_weight is None:
            weight = 1.0
        else:
            weight = sample_weight


        if y is not None:
            row_cnt, _ = get_dimensions(X)
            weight = check_weights(weight, expand_length=row_cnt)
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._train_weight_seen_by_model += weight[i]
                    self._partial_fit(X[i], y[i], self.classes, weight[i])

        return self

    def _partial_fit(self, X, y, classes=None, sample_weight=1.0):
        self.instances_seen += 1

        if self.ensemble is None:
            self._init_ensemble(X)

        for i in range(self.n_estimators):
            y_predicted = self.ensemble[i].predict(np.asarray([X]))
            self.ensemble[i].evaluator.add_result(y_predicted, y, sample_weight)

            k = self._random_state.poisson(self.lambda_value)
            if k > 0:
                self.ensemble[i].partial_fit(np.asarray([X]), np.asarray([y]),
                                             classes=classes,
                                             sample_weight=np.asarray([k]),
                                             instances_seen=self.instances_seen)

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the class labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        y_proba = self.predict_proba(X)
        n_rows = y_proba.shape[0]
        y_pred = np.zeros(n_rows, dtype=int)
        for i in range(n_rows):
            index = np.argmax(y_proba[i])
            y_pred[i] = index
        return y_pred

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Class probabilities are calculated as the mean predicted class probabilities
        per base estimator.

        Parameters
        ----------
         X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the class probabilities.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for all instances in X.
            If class labels were specified in a `partial_fit` call, the order of the columns
            matches `self.classes`.
            If classes were not specified, they are assumed to be 0-indexed.
            Class probabilities for a sample shall sum to 1 as long as at least one estimators
            has non-zero predictions.
            If no estimator can predict probabilities, probabilities of 0 are returned.
        """
        if self.ensemble is None:
            self._init_ensemble(X)

        r, _ = get_dimensions(X)
        y_proba = []
        for i in range(r):
            votes = deepcopy(self.get_votes_for_instance(X[i]))
            if votes == {}:
                # Estimator is empty, all classes equal, default to zero
                y_proba.append([0])
            else:
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes)
                if self.classes is not None:
                    votes_array = np.zeros(int(max(self.classes)) + 1)
                else:
                    votes_array = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    votes_array[int(key)] = value
                y_proba.append(votes_array)
        # Set result as np.array
        if self.classes is not None:
            y_proba = np.asarray(y_proba)
        else:
            # Fill missing values related to unobserved classes to ensure we get a 2D array
            y_proba = np.asarray(list(itertools.zip_longest(*y_proba, fillvalue=0.0))).T
        return y_proba

    def reset(self):
        """Reset ARF."""
        self.ensemble = None
        self.instances_seen = 0
        self._train_weight_seen_by_model = 0.0
        self._random_state = check_random_state(self.random_state)

    def get_votes_for_instance(self, X):
        if self.ensemble is None:
            self._init_ensemble(X)
        combined_votes = {}

        for i in range(self.n_estimators):
            vote = deepcopy(self.ensemble[i].get_votes_for_instance(X))
            if vote != {} and sum(vote.values()) > 0:
                vote = normalize_values_in_dict(vote, inplace=True)
                if not self.disable_weighted_vote:
                    performance = self.ensemble[i].evaluator.accuracy_score()\
                        if self.performance_metric == 'acc'\
                        else self.ensemble[i].evaluator.kappa_score()
                    if performance != 0.0:  # CHECK How to handle negative (kappa) values?
                        for k in vote:
                            vote[k] = vote[k] * performance
                # Add values
                for k in vote:
                    try:
                        combined_votes[k] += vote[k]
                    except KeyError:
                        combined_votes[k] = vote[k]
        return combined_votes

    def _init_ensemble(self, X):
        self._set_max_features(get_dimensions(X)[1])

        self.ensemble = [ARFBaseLearner(index_original=i,
                                        classifier=ARFHoeffdingTreeClassifier(
                                            max_byte_size=self.max_byte_size,
                                            memory_estimate_period=self.memory_estimate_period,
                                            grace_period=self.grace_period,
                                            split_criterion=self.split_criterion,
                                            split_confidence=self.split_confidence,
                                            tie_threshold=self.tie_threshold,
                                            binary_split=self.binary_split,
                                            stop_mem_management=self.stop_mem_management,
                                            remove_poor_atts=self.remove_poor_atts,
                                            no_preprune=self.no_preprune,
                                            leaf_prediction=self.leaf_prediction,
                                            nb_threshold=self.nb_threshold,
                                            nominal_attributes=self.nominal_attributes,
                                            max_features=self.max_features,
                                            random_state=self.random_state),
                                        instances_seen=self.instances_seen,
                                        drift_detection_method=self.drift_detection_method,
                                        warning_detection_method=self.warning_detection_method,
                                        is_background_learner=False)
                         for i in range(self.n_estimators)]

    def _set_max_features(self, n):
        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = round(math.sqrt(n))
        elif self.max_features == 'log2':
            self.max_features = round(math.log2(n))
        elif isinstance(self.max_features, int):
            # Consider 'max_features' features at each split.
            pass
        elif isinstance(self.max_features, float):
            # Consider 'max_features' as a percentage
            self.max_features = int(self.max_features * n)
        elif self.max_features is None:
            self.max_features = n
        else:
            # Default to "auto"
            self.max_features = round(math.sqrt(n))
        # Sanity checks
        # max_features is negative, use max_features + n
        if self.max_features < 0:
            self.max_features += n
        # max_features <= 0
        # (m can be negative if max_features is negative and abs(max_features) > n),
        # use max_features = 1
        if self.max_features <= 0:
            self.max_features = 1
        # max_features > n, then use n
        if self.max_features > n:
            self.max_features = n


class ARFBaseLearner(BaseSKMObject):
    """ARF Base Learner class.

    Parameters
    ----------
    index_original: int
        Tree index within the ensemble.
    classifier: ARFHoeffdingTreeClassifier
        Tree classifier.
    instances_seen: int
        Number of instances seen by the tree.
    drift_detection_method: BaseDriftDetector
        Drift Detection method.
    warning_detection_method: BaseDriftDetector
        Warning Detection method.
    is_background_learner: bool
        True if the tree is a background learner.

    Notes
    -----
    Inner class that represents a single tree member of the forest.
    Contains analysis information, such as the numberOfDriftsDetected.

    """
    def __init__(self,
                 index_original,
                 classifier: ARFHoeffdingTreeClassifier,
                 instances_seen,
                 drift_detection_method: BaseDriftDetector,
                 warning_detection_method: BaseDriftDetector,
                 is_background_learner):
        self.index_original = index_original
        self.classifier = classifier
        self.created_on = instances_seen
        self.is_background_learner = is_background_learner
        self.evaluator_method = ClassificationPerformanceEvaluator

        # Drift and warning
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method

        self.last_drift_on = 0
        self.last_warning_on = 0
        self.nb_drifts_detected = 0
        self.nb_warnings_detected = 0

        self.drift_detection = None
        self.warning_detection = None
        self.background_learner = None
        self._use_drift_detector = True
        self._use_background_learner = True

        self.evaluator = self.evaluator_method()

        # Initialize drift and warning detectors
        if drift_detection_method is not None:
            self._use_drift_detector = True
            self.drift_detection = deepcopy(drift_detection_method)

        if warning_detection_method is not None:
            self._use_background_learner = True
            self.warning_detection = deepcopy(warning_detection_method)

    def reset(self, instances_seen):
        if self._use_background_learner and self.background_learner is not None:
            self.classifier = self.background_learner.classifier
            self.warning_detection = self.background_learner.warning_detection
            self.drift_detection = self.background_learner.drift_detection
            self.evaluator_method = self.background_learner.evaluator_method
            self.created_on = self.background_learner.created_on
            self.background_learner = None
        else:
            self.classifier.reset()
            self.created_on = instances_seen
            self.drift_detection.reset()
        self.evaluator = self.evaluator_method()

    def ss_partial_fit(self, X, sample_weight):
        self.classifier.ss_partial_fit(X, sample_weight)

    def partial_fit(self, X, y, classes, sample_weight, instances_seen):
        self.classifier.partial_fit(X, y, classes=classes, sample_weight=sample_weight)

        if self.background_learner:
            self.background_learner.classifier.partial_fit(X, y,
                                                           classes=classes,
                                                           sample_weight=sample_weight)

        if self._use_drift_detector and not self.is_background_learner:
            correctly_classifies = self.classifier.predict(X) == y
            n_classes = len(self.classifier.return_w())
            weight = 1 / (n_classes * self.classifier.return_w()[y])[0]
            # Check for warning only if use_background_learner is active
            if self._use_background_learner:
                self.warning_detection.add_element(int(not correctly_classifies) * weight)
                # Check if there was a change
                if self.warning_detection.detected_change():
                    self.last_warning_on = instances_seen
                    self.nb_warnings_detected += 1
                    # Create a new background tree classifier
                    background_learner = self.classifier.new_instance()
                    # Create a new background learner object
                    self.background_learner = ARFBaseLearner(self.index_original,
                                                             background_learner,
                                                             instances_seen,
                                                             self.drift_detection_method,
                                                             self.warning_detection_method,
                                                             True)
                    # Update the warning detection object for the current object
                    # (this effectively resets changes made to the object while it
                    # was still a background learner).
                    self.warning_detection.reset()

            # Update the drift detection
            self.drift_detection.add_element(int(not correctly_classifies) * weight)

            # Check if there was a change
            if self.drift_detection.detected_change():
                self.last_drift_on = instances_seen
                self.nb_drifts_detected += 1
                self.reset(instances_seen)


    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def get_votes_for_instance(self, X):
        return self.classifier.get_votes_for_instance(X)
