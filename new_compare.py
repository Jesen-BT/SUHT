import random
import numpy as np
from SS_ARF import SemiAdaptiveRandomForestClassifier
from SS_DWM import SemiDynamicWeightedMajorityClassifier
from SS_LB import SemiLeveragingBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, DynamicWeightedMajorityClassifier, LeveragingBaggingClassifier
from skmultiflow.data.file_stream import FileStream
from multiclass_evaluator import MetricsCalculator
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import OnlineUnderOverBaggingClassifier
from skmultiflow.meta import OnlineSMOTEBaggingClassifier
import pandas as pd
import os

file_name = ['agrawal', 'hyp', 'rbf', 'sea', 'sine', 'stag', 'tree', 'wave', 'weather', 'elec', 'covtype', 'sensor', 'drebin_drift']

# Budget = [0.01, 0.05, 0.1]
Budget = [0.1]

for bg in Budget:
    result_g_mean = []
    g_mean_div = []
    result_f1 = []
    f1_div = []
    for name in file_name:
        file_results_g_mean = []
        file_results_f1 = []
        for _ in range(1):
            SS_ARF = SemiAdaptiveRandomForestClassifier(split_confidence=0.1, n_estimators=20)
            SS_DWM = SemiDynamicWeightedMajorityClassifier(n_estimators=20)
            SS_LB = SemiLeveragingBaggingClassifier(n_estimators=20)
            ARF = AdaptiveRandomForestClassifier(n_estimators=20)
            DWM = DynamicWeightedMajorityClassifier(base_estimator=HoeffdingTreeClassifier(), n_estimators=20)
            LB = LeveragingBaggingClassifier(base_estimator=HoeffdingTreeClassifier(), n_estimators=20)
            SmSCluster = NaiveBayes()
            SSEA = OnlineUnderOverBaggingClassifier(base_estimator=HoeffdingTreeClassifier())
            OSNN = OnlineSMOTEBaggingClassifier(base_estimator=HoeffdingTreeClassifier())
            Models = [SS_ARF, ARF, SS_DWM, DWM, SS_LB, LB, SmSCluster, SSEA, OSNN]

            file = "data/Imbalance_data/" + name + ".csv"
            stream = FileStream(file)
            data, label = stream.next_sample(10)
            classes = stream.target_values
            for i in range(len(Models)):
                Models[i].partial_fit(data, label, classes=classes)

            right = []
            for i in range(len(Models)):
                matrix = MetricsCalculator()
                right.append(matrix)
            data_size = 0

            while stream.has_more_samples():
                new_x, new_y = stream.next_sample()
                data_size = data_size + 1
                for i in range(len(Models)):
                    predict = Models[i].predict(new_x)
                    right[i].add_result(prediction=predict, label=new_y)
                    if random.random() < bg:
                        Models[i].partial_fit(new_x, new_y)
                        if i == 0 or i == 2 or i == 4:
                            Models[i].ss_partial_fit(new_x)
                    else:
                        if i == 0 or i == 2 or i == 4:
                            Models[i].ss_partial_fit(new_x)
            line_g_mean = []
            line_f1 = []
            for i in range(len(Models)):
                line_g_mean.append(right[i].calculate_g_mean())
                line_f1.append(right[i].calculate_balanced_accuracy())
            file_results_g_mean.append(line_g_mean)
            file_results_f1.append(line_f1)
        result_g_mean.append(list(np.mean(np.array(file_results_g_mean), axis=0)))
        result_f1.append(list(np.mean(np.array(file_results_f1), axis=0)))
        g_mean_div.append(list(np.std(np.array(file_results_g_mean), axis=0)))
        f1_div.append(list(np.std(np.array(file_results_f1), axis=0)))
    result_g_mean = pd.DataFrame(result_g_mean)
    result_f1 = pd.DataFrame(result_f1)
    g_mean_div = pd.DataFrame(g_mean_div)
    f1_div = pd.DataFrame(f1_div)
    result_g_mean.columns = ['SSARF', 'ARF', 'SSDWM', 'DWM', 'SSLB', 'LB', 'SmSCluster', 'SSEA', 'OSNN']
    result_f1.columns = ['SSARF', 'ARF', 'SSDWM', 'DWM', 'SSLB', 'LB', 'SmSCluster', 'SSEA', 'OSNN']
    g_mean_div.columns = ['SSARF', 'ARF', 'SSDWM', 'DWM', 'SSLB', 'LB', 'SmSCluster', 'SSEA', 'OSNN']
    f1_div.columns = ['SSARF', 'ARF', 'SSDWM', 'DWM', 'SSLB', 'LB', 'SmSCluster', 'SSEA', 'OSNN']
    result_g_mean.index = file_name
    result_f1.index = file_name
    g_mean_div.index = file_name
    f1_div.index = file_name
    gmean_folder_path = "result/g-mean/"
    if not os.path.exists(gmean_folder_path):
        os.makedirs(gmean_folder_path)
    f1_folder_path = "result/balanced_accuracy/"
    if not os.path.exists(f1_folder_path):
        os.makedirs(f1_folder_path)
    result_g_mean.to_csv(gmean_folder_path + str(bg) + '.csv')
    result_f1.to_csv(f1_folder_path + str(bg) + '.csv')
    # g_mean_div.to_csv(gmean_folder_path + str(bg) + '_SD.csv')
    # f1_div.to_csv(f1_folder_path + str(bg) + '_SD.csv')
    print("finish" + str(bg))