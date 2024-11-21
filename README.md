# FakefaceDetect
Machine learning for real and fake face detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## Project Organization

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

Workflow
1. Set dataset benchmark and filepath in config.yaml, Set param for pre-process & ML in params.yaml
2. Run make_dataset.py to split train test data
3. Run src/prepare.py to pre-processd data into npy array (please run for train and test seperately[define in params.yaml])
4. Run src/train.py to train model with c-v, then output final model performance (include hyperparameter tuning)
5. For evaluate.py which is used to test external dataset (Skip train stage)