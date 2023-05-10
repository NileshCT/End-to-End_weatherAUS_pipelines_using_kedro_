from kedro.pipeline import Pipeline, node

from .nodes import extract_training_data,treat_missing,training_data_split,y_training,lebal_encoder

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=extract_training_data,
                inputs="weather_aus_raw",
                outputs="df1",
                name="extract_training_data_node",
            ),
            node(
                func=treat_missing,
                inputs="df1",
                outputs="df1_treat_training_data",
                name="treat_missing_node",
            ),
            node(
                func=training_data_split,
                inputs="df1_treat_training_data",
                outputs="X_training",
                name="training_data_split_node",

            ),
             node(
                func=y_training,
                inputs=["df1_treat_training_data","X_training"],
                outputs="y_training",
                name="y_training_node",
            ),
            node(
                func=lebal_encoder,
                inputs="X_training",
                outputs="X_training_le",
                name="lebal_encoder_node",
            )
        ]
    ) 
