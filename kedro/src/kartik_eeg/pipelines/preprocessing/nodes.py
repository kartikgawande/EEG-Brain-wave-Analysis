"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.9
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger=logging.getLogger(__name__)
def label_training_stamps(tsdf: pd.DataFrame, labels: dict, y_col: str):
    # print(labels)
    # {'non-conflicting': {'label': 0, 'files': ['1.py', '2.py', '3.py', '4.py']}, 
    #  'conflicting': {'label': 1, 'files': ['3.py']}}
    
    conflicting_files = labels['conflicting']['files']
    conflicting_label = labels['conflicting']['label']
    non_conflicting_files = labels['non-conflicting']['files']
    non_conflicting_label = labels['non-conflicting']['label']

    #labeling
    tsdf[y_col]=""
    tsdf.loc[((tsdf['task']==1)&(tsdf['file'].isin(non_conflicting_files))),y_col]=non_conflicting_label
    tsdf.loc[((tsdf['task']==2)&(tsdf['file'].isin(conflicting_files))),y_col]=conflicting_label
    labeled_tsdf=tsdf[tsdf[y_col]!=""]
    return labeled_tsdf

def raw_to_filtered_without_task_3(dataset: pd.DataFrame, labeledtrainingsStamps: pd.DataFrame, two_min_training_intervals: bool):
    filtered_eeg = pd.DataFrame()
    for user in labeledtrainingsStamps['participant'].unique():
        user_stamps_no_task_3=labeledtrainingsStamps[(labeledtrainingsStamps['participant']==user) & \
                                    (labeledtrainingsStamps['task']!=3)].copy()
        # user_stamps_no_task_3['start']=pd.to_datetime(user_stamps_no_task_3['start'], format='%H:%M:%S')
        # user_stamps_no_task_3['end']=pd.to_datetime(user_stamps_no_task_3['end'], format='%H:%M:%S')
        user_stamps_no_task_3.drop(columns=['participant','task','file'], inplace=True)
        for _, stamp in user_stamps_no_task_3.iterrows():
            start = stamp['start']
            end = stamp['end']
            label = stamp['label']
            # logger.info(f"original start: {start} original end: {end}")

            if not two_min_training_intervals:
                logger.warning("Using actual timestamps.")
                temp_df = dataset[(start<=dataset['ftime']) & (dataset['ftime']<=end) & (dataset['participant']==user)].copy()
            else:
                logger.warning("Using two min interval timestamps.")
                end = start+pd.Timedelta(minutes=1)
                start = start-pd.Timedelta(minutes=1)
                # logger.info(f"start: {start} end: {end}")
                temp_df = dataset[(start<=dataset['ftime']) & (dataset['ftime']<=end) & (dataset['participant']==user)].copy()
            temp_df['label'] = label
            filtered_eeg= pd.concat([filtered_eeg, temp_df])
    filtered_eeg_cols=filtered_eeg.columns.tolist()
    first_cols=['participant','ftime','label']
    filtered_eeg_cols=[col for col in filtered_eeg_cols if col not in first_cols]
    return filtered_eeg[first_cols+filtered_eeg_cols]

def scattered_to_raw(scattered_dataset_path:str,parameters:dict):
    TIME_COL = parameters['time_col']
    SCATTERED_DATASET_PATH = Path(scattered_dataset_path)
    TASK_ONE_FILES = parameters['task_one_files']
    TASK_TWO_FILES = parameters['task_two_files']
    TASK_THREE_FILES = parameters['task_three_files']
    df=pd.DataFrame()
    for i in range(1,31):
        USER_NO=str(i)
        SCATTERED_USER_DATA_PATH = SCATTERED_DATASET_PATH/USER_NO
        TASK_ONE = 'task_one'
        TASK_TWO = 'task_two'
        TASK_THREE = 'task_three'
        df_user = {TASK_ONE: get_merged_df(SCATTERED_USER_DATA_PATH, TASK_ONE_FILES, TIME_COL),
                TASK_TWO: get_merged_df(SCATTERED_USER_DATA_PATH, TASK_TWO_FILES, TIME_COL),
                TASK_THREE: get_merged_df(SCATTERED_USER_DATA_PATH, TASK_THREE_FILES, TIME_COL)}
        df_user = pd.concat([df_user[TASK_ONE],df_user[TASK_TWO],df_user[TASK_THREE]])
        df_user['participant']=USER_NO
        df=pd.concat([df,df_user])
    df['ftime']=pd.to_datetime(df['ftime'])
    df_cols=df.columns.to_list()
    first_columns=['participant','ftime']
    df_cols=[col for col in df_cols if col not in first_columns]
    return df[first_columns+df_cols]
    
def get_merged_df(files_directory, files, merge_col):
    df = pd.DataFrame()
    for file in files:
        df_file=pd.read_csv(os.path.join(files_directory,file), index_col=0)
        if has_more_than_one_obj_col(df_file, merge_col):
            raise Exception(f'More than one obj present in {file}')
        if has_num_col_other_than_float(df_file):
            raise Exception(f'Non float num col present in {file}')
        df_file=replace_int_with_extremes(df_file)
        df_file=df_file.replace(np.nan, 0)
        df_file=df_file.replace(np.inf, 0)
        df_file=df_file.replace(-np.inf, 0)
        if df.shape[1]==0:
            df=df_file
        else:
            df=pd.merge(df,df_file,on=merge_col,suffixes=(None,f'_file_{file}'))
    return df

def has_more_than_one_obj_col(df: pd.DataFrame, merge_col):
    # Select columns that are of object type
    object_cols = df.select_dtypes(include=['object']).columns.to_list()
    return len([col for col in object_cols if col!=merge_col])!=0

def has_num_col_other_than_float(df: pd.DataFrame):
    # Select columns that are of object type
    dtypes_cols = df.dtypes.unique()
    non_float_num_cols = [col for col in dtypes_cols if col!='float64' and col!='O']
    return len(non_float_num_cols)!=0

def replace_int_with_extremes(df:pd.DataFrame):
    for column in df.columns:
                if df[column].dtype != 'object':  # Ensure the column has a numeric type
                    # Replace -inf with the minimum of the column excluding -inf
                    min_val = df[column][df[column] != -np.inf].min()
                    df[column]=df[column].replace(-np.inf, min_val)
                    
                    # Replace inf with the maximum of the column excluding inf
                    max_val = df[column][df[column] != np.inf].max()
                    df[column]=df[column].replace(np.inf, max_val)
    return df


    