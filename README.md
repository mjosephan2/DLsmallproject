# Image Classification of Covid-19 X-Ray Images
50.039 Theory & Practice of Deep Learning - Small Project

## Introduction
This is a project where we have to classify lung xray images - normal, infected(non-covid), infected(covid)  

## Exploration
For a good visual overview of what was developed, training, model, etc., 
you could take a look at these notebooks:  
- *Multi class modelling.ipynb*
- *Two binary class modelling.ipynb*


## Usage
The models weights are saved under /models folder
To use the models for prediction and inference, you could use the model classes created in this model.py
For greater detail of how the functions could be used, you could take a look at this jupyter notebook (Multi class modelling.ipynb) 
where the models are used in greater detail
<br/>
To get you started, here's an example of how you could use the model and reproduce the results for one of the implemented models for instance:
```python
from models import AlexNet
alexNet = load(AlexNet(3), "model/alexNet/alexNet")
ld_val = Lung_Dataset(groups="val")

validloader = DataLoader(ld_val, batch_size = 1, shuffle = True)
acc_alexNet = evaluation(alexNet, validloader)
print("AlexNet accuracy: ", acc_alexNet)

# Display confusion matrix (optional)
from utils import plot_confusion_matrix
cm_alexNet = np.array(confusionMatrix(alexNet, validloader))
plot_confusion_matrix(cm_alexNet, ["normal", "infected covid", "infected non covid"])
```
For more information and detailed notebook, please look at *Multi class modelling.ipynb*