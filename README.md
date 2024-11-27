# Master in Data Science AUEB 2024-2025
## Practical Data Science - Assignment 2
## Stylianos Giagkos f3352410
# Food Hazard Detection Challenge 

This repository contains the solution for the **Food Hazard Detection Challenge**. The challenge involves classifying food safety-related incidents based on short titles and long descriptions. The solution leverages both **Finetuned PubMedBERT** and **LightGBM (LGBM)** models for classification into hazard-category, product-category, hazard, and product classifications.

## Overview

### Project Workflow:
1. **Exploratory Data Analysis (EDA)**: Data cleaning, exploration, and visualization.
2. **Modeling**: Two different approaches are used for classification:
   - **Finetuned PubMedBERT**: Fine-tuned on the dataset to classify food hazard-related texts.
   - **LightGBM**: A gradient boosting model for classification based on features derived from the data.
3. **Evaluation**: Both models are evaluated on performance metrics such as accuracy, precision, recall, and F1-score.
4. **Training and Submission**: Generation of final predictions based on the final optimal model and submission in the required format on CodaLab competition.

### Subtasks (Performed Separately for Title and Text):
#### Subtask 1:
- **Classify hazard-category**: Classifies the general hazard type.
- **Classify product-category**: Classifies the general product type.

#### Subtask 2:
- **Classify hazard**: Classifies the specific hazard type.
- **Classify product**: Classifies the specific product type.

## Repository Files Description:

The following files are included in the repository:

### Submission Notebooks

- **[Submission Model Finetuned PubMedBERT PDS A2 Food Hazard Detection.ipynb]**  
  Submission notebook for a model finetuned with more methods (eg early stopping, weighted crossentropy loss) on the **initial training set** using PubMedBERT.
  The model was initiated from the benchmark model and is developed further in order to make the best submission possible for the **Food Hazard Detection Challenge**.

- **[Submission Model LGBM PDS A2 Food Hazard Detection.ipynb]**  
  Submission notebook for LightGBM models trained on the **initial training set**.

- **[AnB_Submission_PubMedBERT.ipynb]**  
  Submission notebook for a model finetuned on an **augmented training set** using PubMedBERT.


### Benchmarks Notebooks

- **[Benchmark Models Finetuned PubMedBERT PDS A2 Food Hazard Detection.ipynb]**  
  Notebook for running and evaluating benchmark models using PubMedBERT with the **initial training set**.
  The specific version is actually a simplified version in order to investigate the influence of title or text
  as main text feature, in the classification task. The submission model is much more advance fro the scope of techniques in finetuning.
  
- **[Benchmark Models LGBM PDS A2 Food Hazard Detection.ipynb]**  
  Notebook for experimenting with LightGBM models for food hazard detection using the **initial training set**.

- **[AnB_Benchmark_PubMedBERT.ipynb]**  
  Notebook for evaluating benchmark models using an **augmented training set** with PubMedBERT.


### Extra Notebooks
- **[EDA Notebook PDS A2 Food Hazard Detection.ipynb]**  
  Notebook for performing Exploratory Data Analysis (EDA) on the **initial training set**.

- **[data_augmentation.ipynb]**  
  Notebook for applying data augmentation using back translation to incident descriptions and titles, oversampling and undersampling , creating an **augmented training set** for imbalanced classes especially in categories of ```hazard``` and ```product```.


## Requirements
Make sure to install the required dependencies before running the code. You can use the following pip command to install the necessary packages:
```bash
pip install torch lightgbm pandas scikit-learn matplotlib tqdm transformers nltk numpy
```
### Additional Libraries:
- **pandas**
- **re**
- **nltk**
- **scikit-learn**
- **torch**
- **transformers**
- **lightgbm**
- **matplotlib**
- **numpy**

## How to Re-run the Solution

### Step 1: Data Preparation
Ensure that the dataset is correctly placed in the expected directory. Adjust file paths if necessary.

### Step 2: Exploratory Data Analysis
Run `EDA PDS A2 Food Hazard Detection.ipynb` to clean and visualize the data.

### Step 3: Model Training
You can choose between:

- **Finetuned PubMedBERT on initial data**: Run `BENCHMARKS Finetuned PubMedBERT PDS A2 Food Hazard Detection.ipynb`.
- **LightGBM**: Run `BENCHMARKS LGBM PDS A2 Food Hazard Detection .ipynb`.
- **Finetuned PubMedBERT on Augmented and Balanced Dataset**: Run `AnB_Benchmark_PubMedBERT.ipynb`.

### Step 4: Model Evaluation
Both models evaluate accuracy, precision, recall, and F1-score.

### Step 5: Retraining for Submission
Retrain using the submission notebooks to create final submissions for competitions based on the best model of the respective benchmarks notebooks:

- **For PubMedBERT on initial data**: Run `SUBMISSION Finetuned PubMedBERT PDS A2 Food Hazard Detection.ipynb`.
- **For LightGBM**: Run `SUBMISSION LGBM PDS A2 Food Hazard Detection .ipynb`.
- **Finetuned PubMedBERT on Augmented and Balanced Dataset**: Run `AnB_Submission_PubMedBERT.ipynb`.

## Benchmark Results

Benchmark Result have to do with the first versions of 
each model Basic and Advanced, due to computational limitations 
the benchmarks could not be repeated for every optimised version
of each Model. Therefore, Benchmarks actually give a guideline on how to proceed
with title or text as features and adopt good metrics.

**Any other optimised version especially for Finetuning Bert Models
is only investigated on submission level.Therefore their training is based on conclusions
from initial Benchmark models as a forst guideline.**


### Finetuned PubMedBERT Model on initial Data (Title-based):
| Task                        | F1-Score |
|-----------------------------|----------|
| hazard-category (Title)     | 0.8288   |
| product-category (Title)    | 0.7494   |
| hazard (Title)              | 0.5899   |
| product (Title)             | 0.2172   |

### Finetuned PubMedBERT Model on intial data (Text-based):
| Task                        | F1-Score |
|-----------------------------|----------|
| hazard-category (Text)      | 0.9459   |
| product-category (Text)     | 0.7583   |
| hazard (Text)               | 0.8166   |
| product (Text)              | 0.2331   |

### LightGBM Model (Title-based):
| Task                        | F1-Score |
|-----------------------------|----------|
| hazard-category (Title)     | 0.7614   |
| product-category (Title)    | 0.5926   |
| hazard (Title)              | 0.5533   |
| product (Title)             | 0.0798   |

### LightGBM Model (Text-based):
| Task                        | F1-Score |
|-----------------------------|----------|
| hazard-category (Text)      | 0.9129   |
| product-category (Text)     | 0.6682   |
| hazard (Text)               | 0.7671   |
| product (Text)              | 0.0479   |

### Finetuned PubMedBERT Model on augmented and balanced Data (Title-based):

| Task               | F1-Score |
|--------------------|----------|
| hazard-category    | 0.9598   |
| product-category   | 0.9710   |
| hazard             | 0.8381   |
| product            | 0.6066   |

### Finetuned PubMedBERT Model on augmented and balanced data (Text-based):

| Task               | F1-Score |
|--------------------|----------|
| hazard-category    | 0.9566   |
| product-category   | 0.9393   |
| hazard             | 0.8370   |
| product            | 0.5500   |




# Competition Results for Subtask 1 (Hazard-category, Product-category)

Leaderboard: https://codalab.lisn.upsaclay.fr/competitions/19955#results

The following shows the results for hazard-category and product-category tasks for Subtask 1:

| Submission File                         | Score         | Status   | 
|-----------------------------------------|---------------|----------|
| submission_finetuned_PubMedBERT v4.zip	| 0.7127662337	| Finished | 
| AnB Data Finetuned PubMedBERT.zip       | 0.6876231681  | Finished | 
| submission.zip (LGBM)                   | 0.6428057851  | Finished | 

**While the submission notebook introduces notable improvements in training methodology and evaluation fairness, these refinements result in only marginal performance gains in the Codalab contest compared to the benchmarks.**






# Competition Results for Subtask 2 (Hazard, Product)

The following shows the results for hazard and product tasks for Subtask 1:

***But the rating shows for Sub Task 2 a score of 0.3376 for submission_finetuned_PubMedBERT v4.zip***


| Submission File                         | Score         | Status   | 
|-----------------------------------------|---------------|----------|
| submission_finetuned_PubMedBERT v4.zip  | 0.00000       | Finished |
| AnB Data Finetuned PubMedBERT.zip	      | 0.00000       | Finished | 
| submission.zip (LGBM)                   | 0.00000       | Finished | 


# Overfitting Indications from Competition Results

The competition results suggest potential overfitting in the models. The scores for the `submission_finetuned_PubMedBERT.zip` and `AnB Data Finetuned PubMedBERT.zip` file were significantly higher for the training set compared to the test set, which may indicate that the model is overfitting to the training data. To mitigate this, I attempted data augmentation in the **data_augmentation.ipynb** file. The benchmark results using the augmented training data are as follows:

### Collected F1-Scores for Title-Focused Classification (Augmented Data)

### Finetuned PubMedBERT Model on augmented and balanced Data (Title-based):

| Task               | F1-Score |
|--------------------|----------|
| hazard-category    | 0.9598   |
| product-category   | 0.9710   |
| hazard             | 0.8381   |
| product            | 0.6066   |

### Finetuned PubMedBERT Model on augmented and balanced data (Text-based):

| Task               | F1-Score |
|--------------------|----------|
| hazard-category    | 0.9566   |
| product-category   | 0.9393   |
| hazard             | 0.8370   |
| product            | 0.5500   |

The submission file associated with this augmented training approach was **AnB_Submission_PubMedBERT.ipynb**. The leaderboard scores for this submission were:

- **ST1:** 0.6876231681 (`AnB Data Finetuned PubMedBERT.zip`) on 11/23/2024 at 23:11:21
- **ST2:** 0.0 (`AnB Data Finetuned PubMedBERT.zip`) on 11/23/2024 at 23:03:11

Despite the data augmentation efforts, there was no significant improvement in the competition leaderboard scores. For instance, ST1 achieved a score of **0.6876231681**, and the ST2 submission showed discrepancies, with the leaderboard reflecting an approximate score of **0.35**. This suggests that data augmentation did not substantially address the overfitting issue or enhance generalization performance in the competition context. Further investigation and alternative approaches may be required to improve model performance. 

# Explanation for Model Performance Despite Augmented Dataset

I am trying to explain why the augmented dataset did not improve the model's performance in the following points:

1. **Nature of Augmentation**: The augmentation techniques I applied might not have introduced enough diversity or complexity in the data. This could mean the model didn't get new, meaningful examples to improve its ability to generalize.

2. **Overfitting**: Despite using an augmented dataset with  and over/under sampling, the model might still was overfitting to both the original and augmented data. 

**An assumption is that the augmented data is too similar to the original, the model may memorize patterns rather than learning to generalize, leading to poor performance on unseen data.**


# Conclusion
This solution combines advanced NLP (Finetuned PubMedBERT) and traditional machine learning (LightGBM) techniques to classify food hazard data. The solution addresses both general and specific hazard and product classification tasks. Both models were trained and evaluated, with their results summarized above.The submission models are retrained based on the benchmark results to generate the final predictions.
You can re-run the solution by following the provided instructions and reproduce the results with the corresponding datasets and models. 

# References:

- [ChatGPT](https://openai.com/chatgpt): For quick assistance, code help, debugging, and guidance on various data science topics.
- [Stack Overflow](https://stackoverflow.com): For troubleshooting and solutions to coding challenges.
- [ Towards Data Science](https://towardsdatascience.com) and [Medium](https://medium.com/tag/data-science): For articles and tutorials on data science techniques and best practices.
