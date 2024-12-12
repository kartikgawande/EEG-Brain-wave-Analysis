"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(tsne_pipeline)

split=[
    node(
        func=split_data,
        inputs=["dataset","params:split_params","params:y_col"],
        outputs=["training_data", "testing_data"],
        name="split_data"
    )
]

non_agnostic_knn_pipeline=[
    node(
        func=split_data,
        inputs=["dataset","params:split_params","params:y_col"],
        outputs=["training_data", "testing_data"],
        name="split_data"
    ),
    node(
        func=non_agnostic_knn,
        inputs=["training_data","testing_data","params:knn_training_params","params:y_col"],
        outputs="non_agnostic_knn_report",
        name="non_agnostic_random_forest"
    )
]

non_agnostic_random_forest_pipeline=[
    node(
        func=split_data,
        inputs=["dataset","params:split_params","params:y_col"],
        outputs=["training_data", "testing_data"],
        name="split_data"
    ),
    node(
        func=non_agnostic_random_forest,
        inputs=["training_data","testing_data","params:random_forest_training_params","params:y_col"],
        outputs="non_agnostic_random_forest_report",
        name="non_agnostic_random_forest"
    )
]

agnostic_knn_pipeline=[
    node(
        func=agnostic_knn,
        inputs=["dataset","params:knn_training_params","params:y_col","params:split_params"],
        outputs="agnostic_knn_report",
        name="agnostic_knn"
    )
]

agnostic_random_forest_pipeline=[
    node(
        func=agnostic_random_forest,
        inputs=["dataset","params:random_forest_training_params","params:y_col","params:split_params"],
        outputs="agnostic_random_forest_report",
        name="agnostic_random_forest"
    )
]

tsne_pipeline=[
    # node(
    #     func=split_data,
    #     inputs=["dataset","params:split_params","params:y_col"],
    #     outputs=["training_data", "testing_data"],
    #     name="split_data"
    # ),
    node(
        func=tsne,
        inputs=["dataset","interval_dataset","params:sprint_5_labels"],
        outputs=None,
        name="tsne_visualization"
    )
]