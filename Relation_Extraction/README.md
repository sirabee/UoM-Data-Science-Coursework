# Relation Extraction for Scientific Texts

This repository contains code for extracting relationships between entities in scientific texts using two approaches:
1. **Traditional Machine Learning-Based Approach** (SVM, Random Forest, XGBoost)
2. **Deep Learning-Based Approach** (BERT Transformer Model)

## **Project Overview**
Relation extraction is a fundamental task in NLP that involves identifying and classifying semantic relationships between entities in text. This project applies **traditional machine learning models** and **deep learning transformer models** to extract relations from scientific literature.

## Requirements
* Python 3.7+
* Required Libraries:
  ```bash
  pip install pandas spacy numpy scikit-learn xgboost matplotlib seaborn torch transformers
  python -m spacy download en_core_web_md
  ```
* These statements are provided in the notebook.

## Dataset Preparation
* Data should be formatted in JSON format with entity relations.
* Example Format
```json
[
  {
    "filename": "E85-1004",
    "text": "This paper reports a completed stage of ongoing research at the University of York...",
    "entities": [
      {
        "entity_id": "T1",
        "label": "Method",
        "start": 112,
        "end": 131,
        "text": "analytical inverses"
      },
      {
        "entity_id": "T2",
        "label": "OtherScientificTerm",
        "start": 138,
        "end": 164,
        "text": "compositional syntax rules"
      }
    ],
    "relations": [
      {
        "relation_id": "R1",
        "type": "USED-FOR",
        "arg1": "T1",
        "arg2": "T2"
      }
    ]
  }
]
```
* Sample dataset is provided in `dataset.json`.
* Note: If testing a pre-trained ML model, the type field in relations is not required

## **Methods**
We implement **two different approaches** for relation extraction:

### **1. Traditional Machine Learning-Based Approach**
Trains and evaluates three supervised models:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **XGBoost (XGB)**

These models leverage **hand-crafted linguistic features** to enhance relation extraction performance. The extracted features include:
- **Named Entity Recognition (NER) Tags**  
  - Encoded entity types (`entity1_type_encoded`, `entity2_type_encoded`) help distinguish the type of entities (e.g., "Method", "OtherScientificTerm", "Generic").  
- **Part-of-Speech (POS) Tags**  
  - Features `entity1_POS_encoded` and `entity2_POS_encoded` represent the **grammatical category** of each entity (e.g., noun, verb, adjective).  
- **Dependency Parsing Relations**  
  - `dependency_path_encoded` captures **syntactic relationships** between entities in a sentence.  
- **Sentence-Level Structural Features**  
  - `sentence_length`: The number of words in the sentence, providing insight into the complexity of the context.  
  - `word_distance`: The number of words between two entities, useful for assessing how closely related they are.  
  - `span_similarity`: Measures how semantically similar the two entity spans are based on word embeddings.
- **TF-IDF Vectorized Text Features**  
  - Features `tfidf_0` to `tfidf_100` represent **word importance scores** based on Term Frequency-Inverse Document Frequency (TF-IDF).  

#### Usage Guide
The machine learning notebook can be found in the zip folder with the title - Machine_Learning.ipynb. There are two ways to use this pipeline:
##### Training a New Model
If you want to train your own model from scratch, follow these steps:
1. Ensure `dataset.json` is in the same folder, formatted as shown in the [Dataset Preparation](#dataset-preparation) section.
2. Run all sections under "Data Preprocessing".
3. In the "Load the Dataset" section, input - `train`
4. Run all sections under "Model Training".
   - This trains SVM, Random Forest, and XGBoost.
   - The best XGBoost model is saved for future use after grid search CV.

##### Testing a Pre-Trained Model
If you want to test the saved XGBoost model on new data:
1. Ensure `dataset.json` is in the working directory.
   - The format is the same as in [Dataset Preparation](#dataset-preparation), except the `type` field in `relations` is not required.
2. From this link under the 'Machine Learning' folder - [Text Mining Models](https://drive.google.com/drive/folders/1xpm8H42a1fEGVuTvQmrlfzDyCfsX0qEZ?usp=sharing), download xgb_model.pkl, tf_idf_vectorizer.pkl, Standard_Scaler.pkl and label_encoders.pkl to the working directory.
3. Run all sections under "Data Preprocessing".
4. In the "Load the Dataset" section, input - `test`
5. Run all sections under "Model Testing" to predict relations.

Example format for testing (`dataset.json`):
```json
[
  {
    "filename": "E85-1004",
    "text": "This paper reports a completed stage of ongoing research at the University of York...",
    "entities": [
      {
        "entity_id": "T1",
        "label": "Method",
        "start": 112,
        "end": 131,
        "text": "analytical inverses"
      },
      {
        "entity_id": "T2",
        "label": "OtherScientificTerm",
        "start": 138,
        "end": 164,
        "text": "compositional syntax rules"
      }
    ],
    "relations": [
      {
        "relation_id": "R1",
        "arg1": "T1",
        "arg2": "T2"
      }
    ]
  }
]
```
### **2. Deep Learning-Based Approach**
The code implements a pretrained transformer approach (using BERT-base from Hugging Face) for relation extraction. It fine-tunes BERT with an enhanced classifier head, incorporating techniques such as focal loss, class weighting, and stratified sampling to tackle class imbalance. Hyperparameters like a low learning rate, dropout, weight decay, and gradient accumulation are carefully tuned to ensure stable training and effective generalization. Evaluation metrics such as macro F1, precision, and recall are tracked during training to monitor performance and trigger early stopping.

#### Usage Guide
The deep learning notebook can be found in the zip folder with the title - Deep_Learning.ipynb. There are two ways to use this pipeline:

##### Training a New Model
If you want to train the model from scratch, follow these steps:
1. Ensure `dataset.json` is in the same folder.
2. Run all sections step-by-step in the `dl_bert_training.ipynb` file except for "Test with Pre-loaded Model" section.

##### Testing a Pre-Trained Model
If you want to test the saved model on new data:
1. Ensure to download the model through this link under the 'Deep Learning' folder - [Text Mining Models](https://drive.google.com/drive/folders/1xpm8H42a1fEGVuTvQmrlfzDyCfsX0qEZ?usp=sharing) and keep it in the working directory.
2. Input a sentence in the section "Test with Pre-loaded Model" as an `example_sentence`. Strictly follow the instructions and input  sentence with entity tags as shown in the example with [E1] and [E2]:
"[E1]The hippocampus[/E1] is a critical component of the [E2]limbic system[/E2] in the brain."
3. Run all sections under "Test with Pre-loaded Model" to predict relations using the trained model. 

## **Use of Generative AI Tools**
The authors acknowledge the use of ChatGPT model 4o (OpenAI, 2025) to clean and recommend improvements to our codebase, as well as to clarify the application of various methods in implementing learning models, such as executing a BERT-based relation extraction task and interpreting its results. Output is either presented in quotations where referenced directly, or cited where paraphrased in our own words.
