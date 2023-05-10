from kedro.pipeline import Pipeline, node

from .nodes import train_test_split,logReg,pred,r2_score

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_test_split,
                inputs=["X_training_le","y_training"],
                outputs=["X_train","X_test","y_train","y_test"],
                name="train_test_split_node",
            ),
            node(
                func=logReg,
                inputs=["X_train","X_test","y_train","y_test"],
                outputs="logreg",
                name="logReg_node",  
            ),
            node(
                func=pred,
                inputs=["logreg","X_train","X_test"],
                outputs=["y_pred_train","y_pred_test"],
                name="pred_node",  
            ),
            node(
                func=r2_score,
                inputs=["y_train","y_test","y_pred_train","y_pred_test"],
                outputs=["r2_score_train","r2_score_test"],
                name="r2_score_node",
            ),
        ]
    )