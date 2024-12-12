"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node
from .nodes import scattered_to_raw, raw_to_filtered_without_task_3,label_training_stamps


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(func=scattered_to_raw,
             inputs=["params:scattered_dataset_path","parameters"],
             outputs="raw_dataset",
             name='process_scattered'),
        node(func=label_training_stamps,
             inputs=["training_stamps","params:sprint_5_labels","params:y_col"],
             outputs="labeled_training_stamps",
             name='label_training_stamps'),
        node(func=raw_to_filtered_without_task_3,
             inputs=["raw_dataset","labeled_training_stamps","params:two_min_training_intervals"],
             outputs="interval_dataset",
             name='filter_raw')
    ])