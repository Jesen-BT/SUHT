import random
import numpy as np
from SS_HoeffdingTree import SemiHoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier
from Hoefffding_adaptive_tree import HoeffdingAdaptiveTreeClassifier
from skmultiflow.data.file_stream import FileStream
import pandas as pd
import os
from multiclass_evaluator import MetricsCalculator

file_name = ['agrawal', 'hyp', 'rbf', 'sea', 'sine', 'stag', 'tree', 'wave', 'weather', 'elec', 'covtype', 'sensor', 'drebin_drift']

Budget = [0.01, 0.05, 0.1]

for bg in Budget:
    result_g_mean = []
    g_mean_div = []
    result_f1 = []
    f1_div = []
    for name in file_name:
        file_results_g_mean = []
        file_results_f1 = []
        for _ in range(10):
            SUHT = SemiHoeffdingTreeClassifier()
            HT = HoeffdingTreeClassifier(grace_period=25)
            HAT = HoeffdingAdaptiveTreeClassifier(grace_period=25)
            EHT = ExtremelyFastDecisionTreeClassifier(grace_period=25)
            Models = [SUHT, HT, HAT, EHT]

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
                        if i == 0:
                            Models[i].ss_partial_fit(new_x)
                    else:
                        if i == 0:
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
    result_g_mean.columns = ['SUHT', 'HT', 'HAT', 'EHT']
    result_f1.columns = ['SUHT', 'HT', 'HAT', 'EHT']
    g_mean_div.columns = ['SUHT', 'HT', 'HAT', 'EHT']
    f1_div.columns = ['SUHT', 'HT', 'HAT', 'EHT']
    result_g_mean.index = file_name
    result_f1.index = file_name
    g_mean_div.index = file_name
    f1_div.index = file_name
    gmean_folder_path = "result/base_classifier/g-mean/"
    if not os.path.exists(gmean_folder_path):
        os.makedirs(gmean_folder_path)
    f1_folder_path = "result/base_classifier/balanced_accuracy/"
    if not os.path.exists(f1_folder_path):
        os.makedirs(f1_folder_path)
    result_g_mean.to_csv(gmean_folder_path + str(bg) + '.csv')
    result_f1.to_csv(f1_folder_path + str(bg) + '.csv')
    g_mean_div.to_csv(gmean_folder_path + str(bg) + '_SD.csv')
    f1_div.to_csv(f1_folder_path + str(bg) + '_SD.csv')
    print("finish" + str(bg))