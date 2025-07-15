# FakefaceDetect
Machine learning for real and fake face detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# FakefaceDetect

This project investigates real vs. fake face classification using lightweight deep learning models with a focus on balancing predictive performance and computational efficiency. Built on the Real and Fake Face Detection Dataset, it provides a modular pipeline for data preprocessing, model training, evaluation, and deployment.

## 🔍 Key Highlights
- 🔬 Evaluated **MobileNetV3-Large** and **EfficientNet-B0**
- 💡 Input image types: raw RGB and CLAHE-enhanced
- 🧠 Model selection guided by **NetScore** (performance vs. efficiency)
- 🔁 Data augmentation: horizontal flipping, ±30° rotation
- ⚙️ Model training includes hyperparameter tuning and cross-validation

## 🏆 Results
- **EfficientNet-B0**:  
  - Accuracy: 82.9%  
  - F1-score: 0.840  
  - AUC-ROC: 0.896  
- **MobileNetV3-Large** (with augmentation):  
  - F1-score: 0.817  
  - NetScore: **76.268**  
  - FLOPs: 0.225 GFLOPs  
  - Parameters: 4M  

✅ Conclusion: MobileNetV3-Large with CLAHE preprocessing and augmentation offers a robust and efficient solution for real-time deepfake detection, validating **NetScore** as a selection metric for constrained environments.

## 🧱 Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fakefacedetect and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, /graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── make_dataset.py <-split the raw dataset into train test, stored in data/interim
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fakefacedetect a Python module
    │
    │
    ├── prepare.py              <- pre-process data from data/interim 
    ├── train.py                <- train models and return stats with final model
    │
    └── evaluate                <- test model with specific dataset
```

--------

### Workflow
1. Set dataset benchmark and filepath in config.yaml, Set param for pre-process & ML in params.yaml
2. Run make_dataset.py to split train test data
3. Run src/prepare.py to pre-processd data into npy array (please run for train and test seperately[define in params.yaml])
4. Run src/train.py to train model with c-v, then output final model performance (include hyperparameter tuning)
5. For evaluate.py which is used to test external dataset (Skip train stage)
