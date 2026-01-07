import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff

def run_backtest_experiment(df, train_start, train_end, test_start, test_end):
    """
    Trains a model on a past window and evaluates on a future window.
    """
    # 1. Temporal Slicing
    df['date'] = pd.to_datetime(df['date'])
    
    mask_train = (df['date'] >= pd.to_datetime(train_start)) & (df['date'] <= pd.to_datetime(train_end))
    mask_test = (df['date'] >= pd.to_datetime(test_start)) & (df['date'] <= pd.to_datetime(test_end))
    
    train = df[mask_train]
    test = df[mask_test]
    
    if len(train) < 100 or len(test) < 100:
        return None, "Insufficient data for selected range."

    features = ['fico', 'dti', 'loan_age', 'rate_spread', 'current_state', 'balance']
    target = 'target'
    
    # 2. Train Model (Fast configuration for web demo)
    # Note: In a real run, increase n_estimators
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=4, 
        n_estimators=50, 
        max_depth=4
    )
    model.fit(train[features], train[target])
    
    # 3. Predict
    y_true = test[target]
    y_pred = model.predict(test[features])
    
    # 4. Generate Analysis Artifacts
    labels = [0, 1, 2, 3]
    label_names = ['Current', 'Late', 'Default', 'Prepaid']
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Row-Normalization (Recall): P(Predicted | Actual)
    # This approximates the Transition Probability Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle division by zero
    
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    
    return {
        "model": model,
        "cm_norm": cm_norm,
        "report": report,
        "train_n": len(train),
        "test_n": len(test),
        "label_names": label_names
    }, None

def plot_confusion_heatmap(cm_norm, label_names):
    """Creates a Plotly Heatmap for the dashboard."""
    # Round values for display
    z_text = [[str(round(y, 2)) for y in x] for x in cm_norm]
    
    fig = ff.create_annotated_heatmap(
        z=cm_norm, 
        x=label_names, 
        y=label_names, 
        annotation_text=z_text, 
        colorscale='Blues'
    )
    
    fig.update_layout(
        title="Transition Probability Matrix P(Predicted | Actual)",
        xaxis_title="Predicted State (t+1)",
        yaxis_title="Actual State (t+1)"
    )
    return fig