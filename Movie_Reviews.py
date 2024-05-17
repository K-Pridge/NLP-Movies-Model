import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
st.title('Movie Reviews')

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn import set_config
set_config(transform_output='pandas')
# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)
    
# Define the load raw eda data function with caching
@st.cache_data
def load_data(fpath):
    df = pd.read_csv(fpath)
    return df
    
## Loading our training and test data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)
# Load training data from FPATHS
train_data_fpath  = FPATHS['data']['ml']['train']
X_train, y_train = load_Xy_data(train_data_fpath)
# Load test data from FPATHS
test_data_fpath  = FPATHS['data']['ml']['test']
X_test, y_test = load_Xy_data(test_data_fpath)



# Get text to predict from the text input box
X_to_pred = st.text_input("### Enter text to predict here:", 
                          value="I loved the movie.")

# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model
# Load model from FPATHS
model_fpath = FPATHS['models']['clf']
clf_pipe = load_ml_model(model_fpath)

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
def classification_metrics_streamlit(y_true, y_pred, label='',
                           figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f",
                                    class_names=None):
    """Modified version of classification metrics function from Intro to Machine Learning.
    Updates:
    - Reversed raw counts confusion matrix cmap  (so darker==more).
    - Added arg for normalized confusion matrix values_format
    """
    # Get the classification report
    report = classification_report(y_true, y_pred,target_names=class_names)
    
    ## Save header and report
    header = "-"*70
    final_report = "\n".join([header,f" Classification Metrics: {label}", header,report,"\n"])
        
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            display_labels = class_names, # Added display labels
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0]);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            display_labels = class_names, # Added display labels
                                            colorbar=colorbar,
                                            ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()
    return final_report, fig


## To place the 3 checkboxes on top of each other
show_train = st.checkbox("Show training data.",value=True)
show_test = st.checkbox("Show test data.", value=True)
show_model_params =st.checkbox("Show model params.", value=False)
if st.button("Show model evaluation."):
    
    if show_train == True:
        # Display training data results
        y_pred_train = clf_pipe.predict(X_train)
        report_str, conf_mat = classification_metrics_streamlit(y_train, y_pred_train, label='Training Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")
    if show_test == True: 
        # Display the trainin data resultsg
        y_pred_test = clf_pipe.predict(X_test)
        report_str, conf_mat = classification_metrics_streamlit(y_test, y_pred_test, cmap='Reds',label='Test Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")
        
    if show_model_params:
        # Display model params
        st.markdown("####  Model Parameters:")
        st.write(clf_pipe.get_params())
else:
    st.empty()





# Update the function to decode the prediction
def make_prediction(X_to_pred, clf_pipe=clf_pipe):
    # Get Prediction
    pred_class = clf_pipe.predict([X_to_pred])[0]
    # Decode label
    return pred_class
# Trigger prediction with a button
if st.button("Get prediction."):
    pred_class = make_prediction(X_to_pred)
    st.markdown(f"##### Predicted category:  {pred_class}") 
else: 
    st.empty()













    





