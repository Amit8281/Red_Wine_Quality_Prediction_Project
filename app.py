#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import pandas as pd
import gradio as gr

# Load the trained model
model = pickle.load(open('Extra_tree_classifier.pkl', 'rb'))

def wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, sulphates,alcohol):
    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'fixed_acidity': [fixed_acidity],
        'volatile_acidity': [volatile_acidity],
        'citric_acid': [citric_acid],
        'residual_sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free_sulfur_dioxide': [free_sulfur_dioxide],
        'density': [density],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    # Perform the prediction
    prediction = model.predict(data)[0]
    return "GOOD" if prediction == 1 else "NOT GOOD"

# Create the input components
input_components = [
    gr.inputs.Number(label="fixed_acidity"),
    gr.inputs.Number(label="volatile_acidity"),
    gr.inputs.Number(label="citric_acid"),
    gr.inputs.Number(label="residual_sugar"),
    gr.inputs.Number(label="chlorides"),
    gr.inputs.Number(label="free_sulfur_dioxide"),
    gr.inputs.Number(label="density"),
    gr.inputs.Number(label="sulphates"),
    gr.inputs.Number(label="alcohol")
]

# Create the interface
interface = gr.Interface(
    fn=wine_quality,
    inputs=input_components,
    outputs="text",
    title="WINE QUALITY",
    description="WINE QUALITY DETECTION."
)

# Launch the interface
interface.launch()


# In[ ]:




