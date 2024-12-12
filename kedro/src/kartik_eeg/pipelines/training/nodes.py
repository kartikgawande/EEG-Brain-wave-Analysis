"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.9
"""
from collections import defaultdict
import json
import math
from os import makedirs, mkdir
import random
import sys
from turtle import mode
from networkx import non_neighbors
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

logger=logging.getLogger(__name__)
def split_data(dataset: pd.DataFrame, split_params: dict,  y_col:str):
    test_size=split_params["test_size"]
    random_state = 42 if split_params["reproducible_split"] else None
    # return train_test_split(dataset,test_size=test_size,random_state=random_state,stratify=dataset['participant'])

    # Assuming 'col1' and 'col2' are the columns you want to stratify on
    y = pd.get_dummies(dataset[['participant', y_col]])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(msss.split(dataset, y))

    train_data = dataset.iloc[train_idx]
    test_data = dataset.iloc[test_idx]

    # def calculate_marginal_distribution(data, column):
    #     return data[column].value_counts(normalize=True).sort_index()

    # for col in ['participant', y_col]:
    #     print(f"Marginal distribution in {col} for training set:")
    #     print(calculate_marginal_distribution(train_data, col))
    #     print(f"Marginal distribution in {col} for test set:")
    #     print(calculate_marginal_distribution(test_data, col))
    #     print()

    return train_data, test_data


def non_agnostic_random_forest(trainingData: pd.DataFrame,testData: pd.DataFrame,training_params: dict, y_col:str):
    #todo: code for agnoistic model using logo split
    logger.warning(f"Random Forest, Non Agnostic mode.")
    participants=set(trainingData['participant'])
    participants.update(testData['participant'])
    logger.info(f"Participant count:{len(participants)}")
    logger.info(f"Participants:{participants}")
    ng_n_estimators=training_params["ng_n_estimators"]
    random_state = 42 if training_params["reproducible_model"] else None
    report={}
    for participant in participants:
        participantTrainingData=trainingData[trainingData['participant']==participant]
        participantTestData=testData[testData['participant']==participant]
        X_train=participantTrainingData.drop(columns=[y_col,'ftime'])
        y_train=participantTrainingData[y_col] 
        X_test=participantTestData.drop(columns=[y_col,'ftime'])
        y_test=participantTestData[y_col]
        participant_report={}
        datapoints=X_train.shape[0]
        participant_report["Datapoints"]=datapoints
        # for estimator_ratio in estimators_ratio:
        for n_estimator in ng_n_estimators:
            # estimator_ratio=estimator_ratio/100
            # n_estimator=math.floor(datapoints*estimator_ratio)
            model=RandomForestClassifier(n_estimators=n_estimator, random_state=random_state)
            model.fit(X_train,y_train)
            accuracy=accuracy_score(y_test, model.predict(X_test))*100
            participant_report[f"{n_estimator} n_estimators"]=f"Accuracy: {accuracy:2.2f}"
        report[f"Participant {participant}"]=participant_report
        logger.info(f"Participant {participant} done")
    return report

def non_agnostic_knn(trainingData: pd.DataFrame,testData: pd.DataFrame,training_params: dict, y_col:str):
    #todo: code for agnoistic model using logo split
    logger.warning(f"Random Forest, Non Agnostic mode.")
    participants=set(trainingData['participant'])
    participants.update(testData['participant'])
    logger.info(f"Participant count:{len(participants)}")
    logger.info(f"Participants:{participants}")
    neighbors_ratio=training_params["neighbors_ratio"]
    algorithm=training_params["algorithm"]
    weights=training_params["weights"]
    report={}
    for participant in participants:
        # logger.info(f"Random Forest Training for participant {participant}:")
        participantTrainingData=trainingData[trainingData['participant']==participant]
        participantTestData=testData[testData['participant']==participant]
        X_train=participantTrainingData.drop(columns=[y_col,'ftime'])
        y_train=participantTrainingData[y_col] 
        X_test=participantTestData.drop(columns=[y_col,'ftime'])
        y_test=participantTestData[y_col]
        participant_report={}
        datapoints=X_train.shape[0]
        participant_report["Datapoints"]=datapoints
        for neighbor_ratio in neighbors_ratio:
            n_neighbor=math.floor(neighbor_ratio*datapoints)
            model=KNeighborsClassifier(n_neighbors=n_neighbor,algorithm=algorithm, weights=weights)
            model.fit(X_train,y_train)
            accuracy=accuracy_score(y_test, model.predict(X_test))*100
            # logger.info(f"Accuracy with {n_estimator} n_estimators: {accuracy:2.2f}")
            participant_report[f"{n_neighbor} n_neighbors"]=f"Accuracy: {accuracy:2.2f}"
        report[f"Participant {participant}"]=participant_report
        logger.info(f"Participant {participant} done")
    return report

def agnostic_random_forest(dataset: pd.DataFrame,training_params: dict, y_col:str,split_params:dict):
    report_json={}
    logger.warning("Random Forest, Agnostic mode.")
    logger.warning("Agnostic mode: Make sure u pass entire dataset to the trainer.")
    n_estimators=training_params["n_estimators"]
    random_state = 42 if training_params["reproducible_model"] else None
    logo = LeaveOneGroupOut()
    X=dataset.drop(columns=[y_col,'ftime'])
    y=dataset[y_col]
    participants=dataset['participant']
    report=defaultdict(list)
    final_accuraries=[]
    logger.info(f"Participant count:{len(participants.unique())}")
    logger.info(f"Estimators:{n_estimators}")
    logger.info(f"Participants:{participants.unique()}")
    for n_estimator in n_estimators:
        accuracies=[]
        estimator_report={}
        model=RandomForestClassifier(n_estimators=n_estimator, random_state=random_state)
        participants=set(dataset['participant'])
        for participant in participants:
            test_size=split_params["test_size"]
            random_state = 42 if split_params["reproducible_split"] else None
            participant_void_data=dataset[dataset['participant']!=participant]
            participant_data=dataset[dataset['participant']==participant]
            train, test = participant_void_data, participant_data
            X_train=train.drop(columns=[y_col,'ftime'])
            y_train=train[y_col] 
            X_test=test.drop(columns=[y_col,'ftime'])
            y_test=test[y_col]
            datapoints=X_train.shape[0]
            estimator_report["Datapoints"]=datapoints
            model.fit(X_train, y_train)
            accuracy=accuracy_score(y_test, model.predict(X_test))*100
            estimator_report["Participant "+str(participant)]=f"Accuracy: {accuracy:2.2f}"
            accuracies.append(accuracy)
            report[str(participant)].append(accuracy)
        report_json[str(n_estimator)+" Estimators"]=estimator_report
        mean_accuracy=np.mean(accuracies)
        final_accuraries.append(mean_accuracy)
        report_json[str(n_estimator)+" Estimators"]['Mean Accuracy']=f"{mean_accuracy:2.2f}"
        logger.info(f"Estimator {n_estimator} done")
    overall_report={}
    for participant, accuracies in report.items():
        accuracy=np.mean(accuracies)
        overall_report["Participant "+str(participant)]=f"Accuracy: {accuracy:2.2f}"
    overall_report['Mean Accuracy']=f"{np.mean(final_accuraries):2.2f}"
    report_json['Overall']=overall_report
    return report_json

def agnostic_knn(dataset: pd.DataFrame,training_params: dict, y_col:str,split_params:dict):
    report_json={}
    logger.warning("KNN, Agnostic mode.")
    logger.warning("Agnostic mode: Make sure u pass entire dataset to the trainer.")
    n_neighbors=training_params["n_neighbors"]
    algorithm=training_params["algorithm"]
    weights=training_params["weights"]
    logger.info(f"Training with algorithm: {algorithm} aand weights: {weights}")
    logo = LeaveOneGroupOut()
    X=dataset.drop(columns=[y_col,'ftime'])
    y=dataset[y_col]
    participants=dataset['participant']
    logger.info(f"Participant count:{len(participants.unique())}")
    logger.info(f"neighbors_ratio: {n_neighbors}")
    logger.info(f"Participants:{participants.unique()}")
    report=defaultdict(list)
    final_accuraries=[]
    for n_neighbor in n_neighbors:
        accuracies=[]
        neighbor_report={}
        model = KNeighborsClassifier(n_neighbors=n_neighbor,algorithm=algorithm, weights=weights)
        participants=set(dataset['participant'])
        for participant in participants:
            test_size=split_params["test_size"]
            random_state = 42 if split_params["reproducible_split"] else None
            participant_void_data=dataset[dataset['participant']!=participant]
            participant_data=dataset[dataset['participant']==participant]
            train, test = participant_void_data, participant_data
            
            X_train=train.drop(columns=[y_col,'ftime'])
            y_train=train[y_col] 
            X_test=test.drop(columns=[y_col,'ftime'])
            y_test=test[y_col]
            datapoints=X_train.shape[0]
            neighbor_report["Datapoints"]=datapoints

            model.fit(X_train, y_train)
            accuracy=accuracy_score(y_test, model.predict(X_test))*100
            neighbor_report["Participant "+str(participant)]=f"Accuracy: {accuracy:2.2f}"
            accuracies.append(accuracy)
            report[str(participant)].append(accuracy)
        report_json[str(n_neighbor)+" Neighbors"]=neighbor_report
        mean_accuracy=np.mean(accuracies)
        final_accuraries.append(mean_accuracy)
        report_json[str(n_neighbor)+" Neighbors"]['Mean Accuracy']=f"{mean_accuracy:2.2f}"
        logger.info(f"Neighbor {n_neighbor} done")
    overall_report={}
    for participant, accuracies in report.items():
        accuracy=np.mean(accuracies)
        overall_report["Participant "+str(participant)]=f"Accuracy: {accuracy:2.2f}"
    overall_report['Mean Accuracy']=f"{np.mean(final_accuraries):2.2f}"
    report_json['Overall']=overall_report
    return report_json

def tsne(dataset: pd.DataFrame,interval_dataset:pd.DataFrame,labels_params:dict):
    participants=set(dataset['participant'])
    for participant in participants:
        participant_dataset = dataset[dataset['participant']==participant].copy()
        X = participant_dataset.select_dtypes(include=[np.number]) #also excludes 'ftime' col
        X = X.drop(['label','participant'], axis=1)  # Drop other non-feature columns if needed
        logger.info(f"Features for TSNE: {X.columns}")

        conflicting_label = labels_params['conflicting']['label']
        non_conflicting_label = labels_params['non-conflicting']['label']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Configure t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

        # Fit and transform the data
        fig, ax = plt.subplots(figsize=(10, 8))
        X_embedded = tsne.fit_transform(X_scaled)
        labels=participant_dataset['label']
        unique_labels = np.unique(labels)
        for label in unique_labels:
            ix = labels == label
            legend_entry=""
            if label==conflicting_label:
                legend_entry="Conflicting"
            elif label==non_conflicting_label:
                legend_entry="Non-Conflicting"
            ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], label=legend_entry, alpha=0.6)

        # Plotting
        ax.legend(title="Labels")
        plt.title(f't-SNE Visualization for participant {participant}')
        project_path = Path.cwd()
        save_path=project_path/"data"/"08_reporting"/"TSNEs"
        makedirs(save_path,exist_ok=True)
        fig.savefig(save_path/(str(participant)+".png"), format='png', dpi=300)  # Save the figure as a PNG with high resolution
        plt.close(fig)