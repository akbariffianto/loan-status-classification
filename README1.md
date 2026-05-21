# Credit Risk Prediction for Multifinance Industry

## 1. Project Description
In this project, I developed a **machine learning classification model** to predict credit risk for a multifinance company. The main objective is to distinguish between high-risk borrowers (likely to default or "Charged Off") and low-risk borrowers (likely to fully pay their loans). Accurately assessing credit risk is crucial for lending companies to establish a robust, data-driven credit policy, optimize loan approval decisions, and minimize significant financial losses.

The modeling encompasses an imbalanced binary classification problem where the target variable (`loan_status`) is categorized into:
* **Good Credit (1)**: Loans with a final status of `Fully Paid`.
* **Bad Credit (0)**: Loans with a final status of `Charged Off`.

## 2. Business Problem and Objectives
**Business Problem:**
The lending company faces significant challenges in accurately assessing the credit risk of potential borrowers. Inaccuracies in risk assessment can lead to substantial financial losses due to bad loans (defaults). The core business question is: *"How can we accurately predict the credit risk of a borrower based on historical data to minimize potential losses?"*

**Project Objectives:**
1.  **Optimize Business Decisions:** Provide a strong, data-driven foundation for approving or rejecting loan applications.
2.  **Reduce Potential Losses:** Proactively identify high-risk borrowers to avoid loans that have a high probability of being "Charged Off".
3.  Develop a robust Machine Learning model capable of correctly classifying loan applicants using personal, financial, and credit history indicators.

## 3. Solution Pipeline
The project follows the standard data science workflow:
1.  **Business Understanding:** Defining the problem, target, and goals.
2.  **Data Understanding:** Loading the `loan_data_2007_2014.csv` dataset, checking data types, and understanding the significance of each feature (loan amount, interest rate, DTI, etc.).
3.  **Data Preparation:** * Dropping irrelevant identification columns (`id`, `member_id`, `url`).
    * Handling missing values (e.g., filling empty `emp_length` with 'Tidak Bekerja' / Not Working).
    * Feature Engineering: Converting temporal data (e.g., `issue_d`, `earliest_cr_line`) into usable numerical features like `credit_history_length_month`.
4.  **Exploratory Data Analysis (EDA):** Univariate and bivariate analysis, correlation mapping, and Chi-Square tests for categorical variables to select highly relevant features.
5.  **Modeling:** Splitting data (80% train, 20% test), handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique), and applying **PCA** (Principal Component Analysis) for dimensionality reduction.
6.  **Evaluation:** Assessing models based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 4. Technologies and Tools Used
* **Programming Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest, PCA, GridSearchCV/RandomizedSearchCV)
* **Imbalanced Data Handling:** Imbalanced-learn (SMOTE)
* **Model Persistence:** Joblib

## 5. Modeling and Evaluation
Two primary machine learning algorithms were trained and evaluated: **Logistic Regression** (with ElasticNet penalty) and **Random Forest Classifier**. 

### Key Modeling Steps:
* **Feature Selection:** Features with a correlation score of less than 0.1 with the target variable were dropped to reduce noise.
* **Imbalance Handling:** The dataset was naturally imbalanced. **SMOTE** was utilized to oversample the minority class (Bad Credit) to prevent the model from becoming biased towards the majority class.
* **Dimensionality Reduction:** **PCA** was tested to reduce the computational cost while maintaining the explained variance of the dataset.
* **Hyperparameter Tuning:** RandomizedSearchCV was employed to find the best configurations for the models.

### Results:
Both models demonstrated excellent predictive capabilities, especially after applying data augmentation (SMOTE).
* **Random Forest (After Augment/SMOTE)** achieved an **Accuracy of ~98%** and an **ROC-AUC score of ~0.995**.
* **Logistic Regression (After Augment/SMOTE)** achieved an **Accuracy of ~97.4%** and an **ROC-AUC score of ~0.991**.
* **Recall for Charged Off (Bad Loans):** Both models achieved a very high recall (>90%) for the bad credit class, which is exactly what the business needs to avoid financial losses.

## 6. Project Structure
* `FinalTask_IDX Partners_Data Scientist_Akbar Ariffianto.ipynb`: The main Jupyter Notebook containing the full end-to-end analysis, data cleaning, EDA, modeling, and evaluation.
* `*.pkl` files: The exported trained machine learning models (e.g., `smote_rf_nopca.pkl`, `smote_logreg_nopca.pkl`).
* `loan_data_2007_2014.csv`: The raw dataset used for this project (needs to be downloaded separately).

## 7. How to Run the Project
1.  Clone this repository to your local machine.
2.  Ensure you have installed the required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`).
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
3.  Download the dataset (`loan_data_2007_2014.csv`) and place it in the same directory as the notebook.
4.  Open the Jupyter Notebook and Run All cells.
    ```bash
    jupyter notebook "FinalTask_IDX Partners_Data Scientist_Akbar Ariffianto.ipynb"
    ```

## 8. Author
* **Akbar Ariffianto** - Data Scientist Candidate / Final Task ID/X Partners
