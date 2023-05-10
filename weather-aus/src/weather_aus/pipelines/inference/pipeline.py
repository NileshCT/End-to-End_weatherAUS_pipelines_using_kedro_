from kedro.pipeline import Pipeline, node

from .nodes import inference_data,infer_missing_treat,inference_data_split,lebal_encoder_infer,pred_infer,concat_infer

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=inference_data,
                inputs="weather_aus_raw",
                outputs="inference_data",
                name="inference_data_node",
            ),
            node(
                func=infer_missing_treat,
                inputs="inference_data",
                outputs="inference_data_treat",
                name="infer_missing_treat_node",  
            ),
            node(
                func=inference_data_split,
                inputs="inference_data_treat",
                outputs="X_infer",
                name="inference_data_split_node",  
            ),
            node(
                func=lebal_encoder_infer,
                inputs="X_infer",
                outputs="X_infer_le",
                name="lebal_encoder_infer_node",
            ),
             node(
                func=pred_infer,
                inputs=["logreg","X_infer_le"],
                outputs="y_pred_infer",
                name="pred_infer_node",
            ),
             node(
                func=concat_infer,
                inputs=["y_pred_infer","X_infer"],
                outputs="concate_infer",
                name="concat_infer_node",
            ),
        ]
    )