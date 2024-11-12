#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from time import time
from numpy.random import seed
import json
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
from sklearn.linear_model import ElasticNet

# make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def main():

    # read parameter file
    args = sys.argv[1]
    j = open(args)
    parameters = json.load(j)
    j.close()

    # seeds
    my_seed = parameters['seed']
    seed(my_seed)

    # get features and response(s)
    with open(parameters['features'], 'r') as file:
        features = file.read().splitlines()
    response_variables = parameters['responses']
        
    # convert single response variable (string) to list
    if isinstance(response_variables, str):
        response_variables = [response_variables]

    # get train and test data
    data = pd.read_csv(parameters['data_file'], sep="\t", index_col=0)
    print(list(data.index)[:10]) 
    with open(parameters['train_samples'], 'r') as file:
        train_samples = file.read().splitlines()
    with open(parameters['test_samples'], 'r') as file:
        test_samples = file.read().splitlines()

    # filter train and test data and shuffle train data
    train_dataset = data.loc[train_samples, data.columns.isin(features + response_variables)]
    test_dataset = data.loc[test_samples, data.columns.isin(features + response_variables)]
    train_shuffled = train_dataset.sample(frac=1, random_state=my_seed)
    assert train_dataset.shape[0] == len(train_samples)
    assert train_dataset.shape[1] == len(features) + len(response_variables)
    assert test_dataset.shape[0] == len(test_samples)
    assert test_dataset.shape[1] == len(features) + len(response_variables)
    assert train_shuffled.shape[0] == len(train_samples)
    assert train_shuffled.shape[1] == len(features) + len(response_variables)

    # generate in- and outputs for train and test data
    train_inputs = train_shuffled.drop(response_variables, axis=1)
    test_inputs = test_dataset.drop(response_variables, axis=1)
    train_outputs = train_shuffled[response_variables]
    train_inputs_unshuffled = train_dataset.drop(response_variables, axis=1)
    train_outputs_unshuffled = train_dataset[response_variables]
    test_outputs = test_dataset[response_variables]

    # get all possible parameter combinations
    keys, values = zip(*parameters['model_parameters'].items())
    values = [item if isinstance(item, list) else [item] for item in values]
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # iterate over all parameter combinations
    for i in range(len(param_combinations)):
        print(i)
        p = param_combinations[i]

        p['num_inputs'] = train_inputs.shape[1]
        p['num_outputs'] = train_outputs.shape[1]
        p['seed'] = parameters['seed']
        p['result_folder'] = parameters['result_folder'] + '/Parameters' + str(i)

        # train model
        start_time = time()
        model = ElasticNet(alpha=p['alpha'], l1_ratio=p['l1_ratio'], random_state=p['seed'])
        model.fit(train_inputs, train_outputs[response_variables].values.ravel())
        end_time = time()
        duration = (end_time - start_time)

        # compute and format predictions
        p_test = model.predict(test_inputs)
        p_train = model.predict(train_inputs_unshuffled)
        predictions_test = p_test
        predictions_train = p_train
        predictions_test = pd.DataFrame(predictions_test)
        predictions_train = pd.DataFrame(predictions_train)
        predictions_test.columns = ["pred"]
        predictions_train.columns = ["pred"]
        predictions_test.index = test_inputs.index
        predictions_train.index = train_inputs_unshuffled.index

        # add responses to predictions
        predictions_test = pd.concat([predictions_test, test_outputs], axis=1)
        predictions_train = pd.concat([predictions_train, train_outputs_unshuffled], axis=1)

        # compute error measures
        mse_test = mean_squared_error(p_test, test_outputs)
        mse_train = mean_squared_error(p_train, train_outputs_unshuffled)
        mae_test = mean_absolute_error(p_test, test_outputs)
        mae_train = mean_absolute_error(p_train, train_outputs_unshuffled)

        # save results
        if not os.path.exists(p['result_folder']):
            os.makedirs(p['result_folder'])

        predictions_test.to_csv(p['result_folder'] + '/Predictions_Test.txt', sep="\t")
        predictions_train.to_csv(p['result_folder'] + '/Predictions_Train.txt', sep="\t")
        print("Duration\tMSE_train\tMSE_test\tMAE_train\tMAE_test\n" +
              str(duration) + "\t" + str(mse_train) + "\t" + str(mse_test) + "\t" +
              str(mae_train) + "\t" + str(mae_test),
              file=open(p['result_folder'] + '/Duration_trainMSE_testMSE.txt', 'w'))

        with open(p['result_folder'] + '/Parameters.json', 'w') as fp:
            json.dump(p, fp)


if __name__ == "__main__":
    main()


