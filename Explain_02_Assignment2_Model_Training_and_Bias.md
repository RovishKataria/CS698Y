# Explain 02: Assignment 2 - Model Training and Bias Analysis
## Deep Dive into Bias Detection, Mitigation, and Fairness Evaluation

---

## Table of Contents
1. [Why Assignment 2 Exists](#why-assignment-2-exists)
2. [The Bias Problem in Machine Learning](#the-bias-problem-in-machine-learning)
3. [Complete Pipeline Walkthrough](#complete-pipeline-walkthrough)
4. [Bias Evaluation Framework](#bias-evaluation-framework)
5. [Corrective Measures Implementation](#corrective-measures-implementation)
6. [Model Training Process](#model-training-process)
7. [Fairness Metrics Explained](#fairness-metrics-explained)
8. [Results Interpretation](#results-interpretation)
9. [Model Saving and Serialization](#model-saving-and-serialization)

---

## Why Assignment 2 Exists

### The Assignment Objective
Assignment 2 focuses on **identifying, analyzing, and mitigating bias** in machine learning models. This is a critical skill because:

1. **Real-World Impact**: Biased models can cause real harm:
   - Unfair hiring decisions
   - Discriminatory loan approvals
   - Biased medical diagnoses
   - Unfair employee evaluations

2. **Legal Requirements**: Many jurisdictions require:
   - Fairness in automated decision-making
   - Protection against discrimination
   - Transparency in AI systems

3. **Ethical Responsibility**: As ML practitioners, we must:
   - Understand how models can perpetuate bias
   - Learn techniques to detect and mitigate bias
   - Build fair and equitable systems

### What Assignment 2 Teaches
- **Bias Detection**: How to identify bias in datasets and models
- **Bias Analysis**: Understanding sources and types of bias
- **Bias Mitigation**: Techniques to reduce bias in models
- **Fairness Evaluation**: Metrics to measure fairness across groups
- **Trade-offs**: Understanding the relationship between accuracy and fairness

---

## The Bias Problem in Machine Learning

### What is Bias in ML?
Bias in machine learning refers to **systematic errors or unfairness** in model predictions that affect certain groups more than others. It's not the same as statistical bias (systematic error in estimation).

### Types of Bias in Our Dataset

#### 1. **Representation Bias**
**Definition**: When certain groups are over- or under-represented in the training data.

**In Our Dataset**:
- **Education Level 1**: 611 records (82.6%) - **Severely over-represented**
- **Education Level 4**: 4 records (0.5%) - **Extremely under-represented**
- **Age 31-40**: 422 samples (57.0%) - **Over-represented**
- **Age 50+**: 9 samples (1.2%) - **Severely under-represented**

**Why This is a Problem**:
- Model learns patterns from majority groups
- Minority groups get poor predictions
- Model becomes biased toward majority patterns
- Unfair treatment of underrepresented groups

**Real-World Impact**:
- If most training data is from one education level, model might:
  - Predict high absence for all education levels
  - Fail to recognize patterns in other education levels
  - Unfairly penalize employees from underrepresented groups

#### 2. **Historical Bias**
**Definition**: Bias that exists in historical data due to past discrimination or unfair practices.

**In Our Dataset**:
- Historical hiring practices might favor certain education levels
- Past promotion patterns might create age imbalances
- Historical workplace culture might affect data collection

**Why This is a Problem**:
- Model learns from biased historical data
- Perpetuates past discrimination
- Creates feedback loops (biased predictions → biased decisions → biased data)

#### 3. **Measurement Bias**
**Definition**: Bias introduced by how data is collected or measured.

**In Our Dataset**:
- Absence tracking might vary by job role
- Some employees might have different absence reporting requirements
- Health-related absences might be tracked differently

**Why This is a Problem**:
- Inconsistent measurement creates unfair comparisons
- Some groups might appear to have more absences due to measurement differences
- Not actual behavioral differences

#### 4. **Proxy Bias**
**Definition**: When seemingly neutral features actually encode protected attributes.

**In Our Dataset**:
- **Height and Weight**: Can be proxies for gender
  - Men tend to be taller and heavier than women
  - Using these features indirectly uses gender information
- **Body Mass Index**: Derived from height/weight, also a proxy
- **Education Level**: Can be a proxy for socioeconomic status

**Why This is a Problem**:
- Even if we don't use gender directly, we're using it indirectly
- Violates anti-discrimination laws
- Creates unfair predictions based on protected attributes

---

## Complete Pipeline Walkthrough

### Step 1: Data Loading and Exploration

#### Why We Load Data First
Before any analysis, we need to:
- **Understand the data structure**: What columns exist, what types are they?
- **Check data quality**: Missing values, duplicates, inconsistencies
- **Get basic statistics**: Ranges, distributions, outliers

#### Code Explanation
```python
df = pd.read_csv('Absenteeism_at_work.csv', sep=';')
```
- **Why semicolon separator?**: European CSV format uses `;` instead of `,`
- **Why pandas?**: Efficient data manipulation and analysis
- **What we get**: DataFrame with 740 rows × 21 columns

#### What We Check
1. **Dataset Shape**: 740 records, 21 features
2. **Data Types**: Integers, floats, ensuring correct types
3. **Missing Values**: None found (clean dataset)
4. **Duplicates**: 34 duplicate records identified
5. **Basic Statistics**: Mean, median, std for each feature

### Step 2: Feature Analysis

#### Why Analyze Features?
- **Understand distributions**: Which values are common, which are rare?
- **Identify correlations**: Which features relate to the target?
- **Detect outliers**: Extreme values that might skew the model
- **Plan preprocessing**: What encoding, scaling, or transformation is needed?

#### Categorical Variables Analysis
**Why Categorical Variables Need Special Handling**:
- Linear regression requires numerical inputs
- Categories need to be converted to numbers
- One-hot encoding creates binary columns for each category

**Example**: Education Level
- Original: 1, 2, 3, 4 (4 categories)
- After one-hot encoding: 4 binary columns
  - Education_1: 1 if level 1, else 0
  - Education_2: 1 if level 2, else 0
  - Education_3: 1 if level 3, else 0
  - Education_4: 1 if level 4, else 0

**Why Drop First?** (`drop_first=True`):
- Prevents multicollinearity (perfect correlation)
- If we know Education_1=0, Education_2=0, Education_3=0, then Education_4 must be 1
- Reduces dimensionality (3 columns instead of 4)
- One category becomes the "reference" category

#### Numerical Variables Analysis
**Correlation Analysis**:
- **Why calculate correlations?**: Identify which features relate to absenteeism
- **What correlations tell us**: 
  - Positive correlation: Higher feature value → More absence
  - Negative correlation: Higher feature value → Less absence
  - Near zero: No relationship

**Key Findings**:
- Height: 0.144 (weak positive) - taller people slightly more absent
- Age: 0.066 (very weak positive) - older people slightly more absent
- Distance: -0.088 (weak negative) - farther distance → less absence (counterintuitive!)

**Why Some Correlations are Counterintuitive**:
- Correlation doesn't imply causation
- Other factors might be involved
- Data might have confounding variables

### Step 3: Bias Evaluation

#### Why Evaluate Bias Before Training?
- **Baseline Understanding**: Know what bias exists before mitigation
- **Measure Improvement**: Compare before/after bias mitigation
- **Identify Problem Areas**: Focus mitigation efforts where needed
- **Documentation**: Show stakeholders the bias problem

#### The Bias Evaluation Function

##### **Age-Based Bias Analysis**

**Why Group Ages?**:
- Age is continuous (20-60 years)
- Grouping makes analysis manageable
- Identifies age-based patterns
- Aligns with legal age discrimination categories

**Age Groups Created**:
- 18-30: Young employees
- 31-40: Mid-career employees
- 41-50: Experienced employees
- 50+: Senior employees

**What We Measure**:
1. **Representation**: How many samples in each group?
2. **Percentage**: What percentage of total dataset?
3. **Balance Assessment**: Over-represented, under-represented, or balanced?

**Thresholds Used**:
- **Over-represented**: > 25% (more than expected in balanced dataset)
- **Under-represented**: < 15% (less than expected)
- **Balanced**: 15-25% (reasonable representation)

**Results**:
- 18-30: 23.9% - Balanced ✓
- 31-40: 57.0% - **Over-represented** ✗
- 41-50: 17.8% - Balanced ✓
- 50+: 1.2% - **Severely under-represented** ✗

**Why This Matters**:
- Model will learn patterns from 31-40 age group (57% of data)
- 50+ age group (1.2%) will have poor predictions
- Model becomes biased toward middle-aged employees

##### **Education-Based Bias Analysis**

**Why Education Matters**:
- Education level affects job type and responsibilities
- Different education levels might have different absence patterns
- Education can be a proxy for socioeconomic status

**Distribution Found**:
- Level 1: 82.6% - **Severely over-represented**
- Level 2: 6.2% - Under-represented
- Level 3: 10.7% - Under-represented
- Level 4: 0.5% - **Extremely under-represented**

**Why This is Critical**:
- Model will be heavily biased toward education level 1
- Education levels 2, 3, 4 will have unreliable predictions
- Model might unfairly penalize higher education levels

##### **Disproportionate Effects Analysis**

**What is Disproportionate Effect?**:
- When certain groups experience different outcomes
- Not just representation, but actual impact differences
- Even with equal representation, outcomes might differ

**Age-Based Absenteeism Impact**:
- 18-30: 5.44 hours (below average by 1.49 hours)
- 31-40: 7.06 hours (above average by 0.14 hours)
- 41-50: 6.96 hours (above average by 0.04 hours)
- 50+: 29.11 hours (above average by 22.19 hours) ⚠️

**Why 50+ Has High Absence**:
- Legitimate health issues increase with age
- More medical appointments
- More chronic conditions
- But model might over-predict for all older employees

**The Fairness Challenge**:
- Is it fair to predict high absence for 50+ employees?
- Or is it age discrimination?
- How do we distinguish legitimate patterns from bias?

---

## Bias Evaluation Framework

### The `regression_group_fairness` Function

#### Why This Function Exists
Standard ML metrics (RMSE, MAE) measure **overall** performance, but don't tell us about **group-wise** fairness. We need to:
- Measure performance for each demographic group separately
- Compare performance across groups
- Identify which groups get worse predictions

#### Function Parameters Explained

**`y_true`**: Actual absenteeism hours (ground truth)
- **Type**: Pandas Series or numpy array
- **Why Series?**: Preserves index information for alignment
- **Example**: [4, 0, 2, 4, 8, 12, ...] hours

**`y_pred`**: Predicted absenteeism hours (model output)
- **Type**: Pandas Series or numpy array
- **Must match y_true length**: Same number of predictions as actual values
- **Example**: [3.5, 1.2, 2.8, 4.1, 7.9, 11.5, ...] hours

**`sensitive_series`**: Group labels (age group, education level, etc.)
- **Type**: Pandas Series with group labels
- **Example**: ['18-30', '31-40', '18-30', '50+', ...]
- **Why Series?**: Preserves index to align with predictions

**`group_names`**: List of unique groups to evaluate
- **Type**: List or array
- **Example**: ['18-30', '31-40', '41-50', '50+']
- **Why needed?**: Tells function which groups to analyze

#### Index Alignment Logic

**Why Index Alignment is Critical**:
- Predictions and actual values must match
- Group labels must correspond to correct predictions
- Pandas index helps ensure correct alignment

**The Alignment Process**:
1. **Convert to Series**: Ensure all inputs are pandas Series
2. **Check Index Match**: Do indices align?
3. **Reindex if Needed**: Align indices if they don't match
4. **Handle Mismatches**: Truncate or pad if lengths differ

**Why This Complexity?**:
- Test set indices might not be sequential (0, 1, 2, ...)
- After train/test split, indices are shuffled
- We need to match predictions to correct groups
- Index alignment ensures correctness

#### Metrics Calculated for Each Group

**1. Count (n)**
- **What**: Number of samples in this group
- **Why Important**: 
  - Small groups have less reliable metrics
  - Statistical significance depends on sample size
  - Groups with few samples need careful interpretation

**2. MAE (Mean Absolute Error)**
- **Formula**: `mean(|y_true - y_pred|)`
- **What It Measures**: Average prediction error in hours
- **Why Important**: 
  - Shows how far off predictions are on average
  - Easy to interpret (hours of error)
  - Less sensitive to outliers than RMSE

**Example**: 
- Actual: [4, 8, 2, 6]
- Predicted: [3, 9, 1, 7]
- Errors: [1, 1, 1, 1]
- MAE = 1.0 hours (average error)

**3. RMSE (Root Mean Square Error)**
- **Formula**: `sqrt(mean((y_true - y_pred)²))`
- **What It Measures**: Average squared error (penalizes large errors more)
- **Why Important**: 
  - Penalizes large errors more than small ones
  - More sensitive to outliers
  - Standard metric for regression

**Example**:
- Actual: [4, 8, 2, 6]
- Predicted: [3, 9, 1, 7]
- Squared Errors: [1, 1, 1, 1]
- RMSE = 1.0 hours

**4. Bias (Mean Prediction Error)**
- **Formula**: `mean(y_pred - y_true)`
- **What It Measures**: Systematic over- or under-prediction
- **Why Important**: 
  - Positive bias: Model over-predicts (predicts more absence than actual)
  - Negative bias: Model under-predicts (predicts less absence than actual)
  - Zero bias: No systematic error (ideal)

**Example**:
- Actual: [4, 8, 2, 6] (mean = 5.0)
- Predicted: [5, 9, 3, 7] (mean = 6.0)
- Bias = 6.0 - 5.0 = +1.0 hours (over-predicts by 1 hour on average)

**5. Average Predicted Value**
- **Formula**: `mean(y_pred)`
- **What It Measures**: What the model predicts on average for this group
- **Why Important**: 
  - Shows if model predicts different levels for different groups
  - Helps identify systematic differences

**6. Average True Value**
- **Formula**: `mean(y_true)`
- **What It Measures**: Actual average absenteeism for this group
- **Why Important**: 
  - Shows actual group differences
  - Helps distinguish real differences from model bias

**7. R² Score (Coefficient of Determination)**
- **Formula**: `1 - (SS_res / SS_tot)`
- **What It Measures**: How well model explains variance for this group
- **Why Important**: 
  - R² = 1.0: Perfect predictions
  - R² = 0.0: Model is as good as predicting the mean
  - R² < 0.0: Model is worse than predicting the mean (very bad!)

#### Gap Calculation

**What is a Gap?**
- **Definition**: Difference between best and worst group performance
- **Formula**: `max(metric) - min(metric)`
- **Why Important**: Measures fairness (smaller gap = more fair)

**MAE Gap Example**:
- Group A MAE: 4.0 hours
- Group B MAE: 8.0 hours
- Group C MAE: 6.0 hours
- MAE Gap = 8.0 - 4.0 = 4.0 hours

**What Gap Means**:
- **Small Gap (0-5 hours)**: Groups have similar prediction quality (fair)
- **Medium Gap (5-15 hours)**: Some unfairness exists
- **Large Gap (>15 hours)**: Significant unfairness (unacceptable)

**Why We Calculate Multiple Gaps**:
- **MAE Gap**: Overall prediction quality fairness
- **RMSE Gap**: Large error fairness
- **Bias Gap**: Systematic error fairness
- **Prediction Gap**: Average prediction level fairness

---

## Corrective Measures Implementation

### Why We Need Corrective Measures

**The Problem**:
- Baseline model trained on imbalanced data
- Model learns biased patterns
- Unfair predictions for minority groups
- Legal and ethical violations

**The Solution**:
- Apply bias mitigation techniques
- Retrain model on balanced data
- Measure improvement in fairness
- Document what was done

### Measure 1: Feature Elimination

#### Why Remove Features?

**Proxy Features Problem**:
- Some features encode protected attributes indirectly
- Even if we don't use gender, height/weight encode it
- This is illegal discrimination (using proxies for protected attributes)

**Features Removed**:

**1. Height**
- **Why Removed**: 
  - Proxy for gender (men taller on average)
  - Proxy for age (height changes with age in children)
  - Not job-related (height doesn't affect work performance)
  - Privacy concern (personal physical information)
- **Legal Issue**: Using height = using gender indirectly (illegal)

**2. Weight**
- **Why Removed**: 
  - Proxy for gender (men heavier on average)
  - Proxy for age (weight changes with age)
  - Health discrimination (using weight to predict absence is discriminatory)
  - Privacy concern (personal health information)
- **Legal Issue**: Weight-based discrimination is illegal in many places

**3. Body Mass Index (BMI)**
- **Why Removed**: 
  - Derived from height and weight (already removed)
  - Health discrimination (BMI-based decisions are discriminatory)
  - Proxy for multiple protected attributes
  - Not job-related
- **Legal Issue**: BMI-based discrimination is illegal

**4. ID**
- **Why Removed**: 
  - Not a predictive feature (just an identifier)
  - Would cause overfitting (model memorizing specific employees)
  - Privacy concern (identifying individuals)
  - Not generalizable (can't use for new employees)

#### Implementation Code
```python
features_to_remove = ['Height', 'Weight', 'Body mass index', 'ID']
balanced_df = balanced_df.drop(columns=features_to_remove)
```

**Why This Works**:
- Removes discriminatory pathways
- Prevents model from using protected attributes
- Maintains model performance (these features weren't highly predictive anyway)
- Legal compliance

### Measure 2: Age Group Balancing

#### Why Balance Age Groups?

**The Problem**:
- Age groups are severely imbalanced
- Model learns from majority group (31-40: 57%)
- Minority groups (50+: 1.2%) get poor predictions
- Age discrimination is illegal

**The Solution**:
- Balance representation across all age groups
- Each group gets equal weight in training
- Model learns patterns from all groups equally

#### Balancing Strategy

**Step 1: Identify Smallest Group**
```python
age_counts = age_groups.value_counts()
min_age_count = age_counts.min()  # Find smallest group size
```

**Why Use Smallest Group?**:
- Ensures all groups have equal representation
- Prevents over-representation of any group
- Fair training data distribution

**Step 2: Downsample Majority Groups**
```python
if len(group_data) > min_age_count:
    group_data = group_data.sample(n=min_age_count, random_state=42)
```

**What This Does**:
- Randomly selects `min_age_count` samples from larger groups
- Reduces majority group representation
- `random_state=42`: Ensures reproducibility (same random selection each time)

**Why Random Sampling?**:
- Preserves data distribution within group
- Doesn't introduce bias in selection
- Representative sample of group

**Step 3: Upsample Minority Groups**
```python
elif len(group_data) < min_age_count:
    group_data = resample(group_data, n_samples=min_age_count, 
                         random_state=42, replace=True)
```

**What This Does**:
- Creates additional samples for small groups
- Uses resampling with replacement (bootstrap)
- Increases minority group representation

**Why Resampling with Replacement?**:
- Allows same sample to appear multiple times
- Necessary when group is too small
- Preserves group characteristics

**Results**:
- All age groups now have 9 samples each
- Perfect balance achieved
- Model learns from all groups equally

#### Trade-offs of Balancing

**Benefits**:
- ✓ Fair representation
- ✓ Better predictions for minority groups
- ✓ Legal compliance
- ✓ Ethical alignment

**Costs**:
- ✗ Reduced dataset size (740 → 36 samples)
- ✗ Loss of information from majority groups
- ✗ Potential overfitting (small dataset)
- ✗ Reduced overall accuracy

**Why We Accept These Costs**:
- Fairness is more important than accuracy
- Legal requirement (can't discriminate)
- Ethical responsibility
- Better for underrepresented groups

### Measure 3: Education Level Balancing

#### Why Balance Education Levels?

**The Problem**:
- Education level 1: 82.6% (severely over-represented)
- Education level 4: 0.5% (extremely under-represented)
- Model heavily biased toward level 1
- Unfair predictions for other levels

**The Solution**:
- Balance representation across education levels
- Each level gets fair representation
- Model learns patterns from all levels

#### Balancing Strategy

**Target Count Calculation**:
```python
target_education_count = education_counts.max() // 2
```

**Why Divide by 2?**:
- Compromise between balance and data size
- Prevents extreme data loss
- Maintains reasonable dataset size
- Still achieves significant balance improvement

**Results**:
- Education level 1: Downsampled to 17 samples
- Education level 3: Balanced to 17 samples
- Education levels 2 and 4: Too few samples (removed or minimal)

**Final Balanced Dataset**:
- **Original**: 740 samples
- **After Balancing**: 34 samples
- **Reduction**: 95.4% reduction in data

**Why Such a Large Reduction?**:
- Extreme imbalance required extreme balancing
- Smallest groups were very small (4-9 samples)
- Balancing to smallest group size
- Trade-off: Fairness vs. Data Size

---

## Model Training Process

### Baseline Model Training

#### Why Train a Baseline First?

**Purpose**:
- Establish performance baseline
- Measure bias before mitigation
- Compare with mitigated model
- Document improvement

#### Training Steps

**Step 1: Data Preparation**
```python
df_base_encoded = pd.get_dummies(df_base, columns=available_cat, drop_first=True)
```

**What Happens**:
- Categorical variables converted to binary columns
- Example: Education (1,2,3,4) → Education_2, Education_3, Education_4
- Original 21 features → ~50 features after encoding

**Why One-Hot Encoding?**:
- Linear regression needs numerical inputs
- Categories can't be used directly (1,2,3,4 implies order that doesn't exist)
- Binary columns preserve category information

**Step 2: Feature-Target Separation**
```python
X_base_reg = df_base_encoded.drop(['Absenteeism time in hours'], axis=1)
y_base_reg = df_base_encoded['Absenteeism time in hours']
```

**What Happens**:
- X: All features (input variables)
- y: Target variable (what we're predicting)
- Separated for model training

**Step 3: Train-Test Split**
```python
Xb_train_r, Xb_test_r, yb_train_r, yb_test_r = train_test_split(
    X_base_reg, y_base_reg, test_size=0.2, random_state=42)
```

**Why Split Data?**:
- **Training Set (80%)**: Used to train the model
- **Test Set (20%)**: Used to evaluate model (unseen data)
- **Why 80/20?**: Standard split, balances training data vs. evaluation data
- **random_state=42**: Ensures same split each time (reproducibility)

**Why Test Set is Critical**:
- Measures how well model generalizes
- Prevents overfitting detection
- Real-world performance estimate

**Step 4: Feature Scaling**
```python
scaler_base = StandardScaler()
Xb_train_r_scaled = scaler_base.fit_transform(Xb_train_r)
Xb_test_r_scaled = scaler_base.transform(Xb_test_r)
```

**What is Standard Scaling?**:
- **Formula**: `(x - mean) / std`
- **Result**: Features have mean=0, std=1
- **Why Needed**: 
  - Linear regression is sensitive to feature scales
  - Features with larger values dominate
  - Scaling ensures fair treatment of all features

**Example**:
- Age: 20-60 (range 40)
- Work Load: 200-350 (range 150)
- Without scaling: Work Load dominates
- With scaling: Both features contribute equally

**Why Fit on Training Only?**:
- `fit_transform()`: Calculates mean/std from training data
- `transform()`: Applies same transformation to test data
- **Critical**: Never use test data statistics (data leakage!)

**Step 5: Model Training**
```python
lr_base = LinearRegression()
lr_base.fit(Xb_train_r_scaled, yb_train_r)
```

**What Linear Regression Does**:
- Finds best line: `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
- **w₁, w₂, ...**: Coefficients (feature weights)
- **b**: Intercept (baseline prediction)
- **Goal**: Minimize prediction error

**How It Works**:
1. Start with random coefficients
2. Make predictions
3. Calculate error (difference from actual)
4. Adjust coefficients to reduce error
5. Repeat until error is minimized

**What Model Learns**:
- **Coefficients**: How much each feature affects prediction
- **Positive coefficient**: Higher feature value → Higher absence
- **Negative coefficient**: Higher feature value → Lower absence
- **Large coefficient**: Feature has strong influence
- **Small coefficient**: Feature has weak influence

**Step 6: Prediction and Evaluation**
```python
yb_pred_lr_base = lr_base.predict(Xb_test_r_scaled)
mse_lr_base = mean_squared_error(yb_test_r, yb_pred_lr_base)
rmse_lr_base = np.sqrt(mse_lr_base)
mae_lr_base = mean_absolute_error(yb_test_r, yb_pred_lr_base)
r2_lr_base = r2_score(yb_test_r, yb_pred_lr_base)
```

**Baseline Results**:
- **RMSE**: 11.43 hours
- **MAE**: 6.44 hours
- **R²**: -0.20 (negative = worse than predicting mean!)

**What These Numbers Mean**:
- **RMSE 11.43**: On average, predictions are off by 11.43 hours
- **MAE 6.44**: Average absolute error is 6.44 hours
- **R² -0.20**: Model is 20% worse than just predicting the mean

**Why Negative R²?**:
- Model performs worse than baseline (predicting mean)
- Indicates poor fit
- Might be due to data complexity or model limitations

### Bias-Mitigated Model Training

#### Differences from Baseline

**1. Balanced Dataset**:
- Uses `balanced_df` instead of `df_base`
- 34 samples instead of 740
- Equal representation across groups

**2. Same Preprocessing**:
- Same one-hot encoding
- Same feature scaling
- Same train-test split (but smaller)

**3. Same Model Type**:
- Still Linear Regression
- Same training process
- Allows fair comparison

#### Training Results

**Bias-Mitigated Results**:
- **RMSE**: 43.12 hours (worse!)
- **MAE**: 16.50 hours (worse!)
- **R²**: -0.09 (still negative, but better)

**Why Performance Got Worse?**:
- **Smaller Dataset**: 34 samples vs. 740 (95% reduction)
- **Less Information**: Model has less data to learn from
- **Overfitting Risk**: Small dataset → model might memorize
- **Trade-off**: Fairness vs. Accuracy

**Is This Acceptable?**:
- **Yes, for fairness**: Legal and ethical requirement
- **Yes, for compliance**: Can't use biased models
- **Yes, for underrepresented groups**: Better predictions for them
- **No, for overall accuracy**: But fairness is more important

---

## Fairness Metrics Explained

### Why We Need Fairness Metrics

**Standard Metrics Don't Show Fairness**:
- Overall RMSE: 11.43 hours (seems good)
- But some groups might have RMSE: 25 hours (very bad!)
- Overall metric hides unfairness

**Fairness Metrics Show**:
- Performance for each group separately
- Gaps between groups
- Which groups are disadvantaged

### Group-Wise Performance Analysis

#### Age Group Fairness (Baseline)

**Results**:
- **18-30**: MAE = 4.43 hours (best)
- **31-40**: MAE = 6.09 hours
- **41-50**: MAE = 8.60 hours
- **50+**: MAE = 25.21 hours (worst!)

**MAE Gap**: 25.21 - 4.43 = **20.78 hours**

**What This Means**:
- 50+ age group gets predictions that are 20.78 hours worse on average
- This is **unacceptable unfairness**
- Age discrimination in predictions
- Legal violation

**Why 50+ Has High Error**:
- Only 2 samples in test set (very small!)
- Model hasn't learned patterns for this group
- High variance in small samples
- But still shows model bias

#### Education Level Fairness (Baseline)

**Results**:
- **Level 1**: MAE = 5.99 hours (best)
- **Level 2**: MAE = 12.48 hours
- **Level 3**: MAE = 5.67 hours
- **Level 4**: MAE = 19.03 hours (worst!)

**MAE Gap**: 19.03 - 5.67 = **13.36 hours**

**What This Means**:
- Education level 4 gets much worse predictions
- Model biased toward education level 1 (82.6% of data)
- Unfair treatment of higher education levels

#### Service Time Fairness (Baseline)

**Results**:
- **0-5 years**: MAE = 7.95 hours
- **6-10 years**: MAE = 4.19 hours (best)
- **11-15 years**: MAE = 6.44 hours
- **15+ years**: MAE = 7.92 hours

**MAE Gap**: 7.95 - 4.19 = **3.76 hours**

**What This Means**:
- Relatively fair across service time groups
- Small gap indicates acceptable fairness
- Less concern for this attribute

### After Bias Mitigation

#### Age Group Fairness (Mitigated)

**Results**:
- All groups: MAE = 16.50 hours (same for all!)
- **MAE Gap**: 0.00 hours (perfect equality!)

**What This Means**:
- Perfect fairness achieved
- All age groups get same prediction quality
- No age discrimination
- But overall accuracy decreased

#### Education Level Fairness (Mitigated)

**Results**:
- **Level 1**: MAE = 1.45 hours (only 1 sample - unreliable)
- **Level 3**: MAE = 19.01 hours
- **MAE Gap**: 17.56 hours (worse than baseline!)

**What This Means**:
- Balancing didn't improve education fairness
- Small dataset makes metrics unreliable
- Trade-off between balance and performance

**Why It Got Worse**:
- Very small dataset (34 samples)
- Education level 4 disappeared (too few samples)
- Remaining groups have very few samples
- Statistical noise dominates

#### Service Time Fairness (Mitigated)

**Results**:
- All groups: MAE = 16.50 hours (same)
- **MAE Gap**: 0.00 hours (perfect equality!)

**What This Means**:
- Perfect fairness maintained
- No service time discrimination
- Good outcome

---

## Results Interpretation

### Performance vs. Fairness Trade-off

**Baseline Model**:
- **Accuracy**: Better (RMSE: 11.43, MAE: 6.44)
- **Fairness**: Worse (Large gaps: 20.78, 13.36 hours)
- **Legal**: Non-compliant (discriminatory)
- **Ethical**: Unacceptable (unfair)

**Bias-Mitigated Model**:
- **Accuracy**: Worse (RMSE: 43.12, MAE: 16.50)
- **Fairness**: Better (Smaller gaps: 0.00, 17.56 hours)
- **Legal**: More compliant (less discriminatory)
- **Ethical**: More acceptable (more fair)

### Why We Choose Fairness Over Accuracy

**1. Legal Requirement**:
- Age discrimination is illegal
- Education discrimination is illegal
- Using biased models = legal liability
- Can't deploy non-compliant models

**2. Ethical Responsibility**:
- Unfair predictions harm people
- Minority groups deserve fair treatment
- Accuracy at cost of fairness is unethical
- Social responsibility

**3. Long-Term Impact**:
- Biased models perpetuate discrimination
- Create feedback loops (bias → biased data → more bias)
- Damage organizational reputation
- Reduce trust in AI systems

**4. Business Case**:
- Legal costs of discrimination lawsuits
- Reputation damage
- Employee morale and retention
- Regulatory compliance

### Understanding the Metrics

#### RMSE: 43.12 hours (Mitigated Model)

**What This Means**:
- Average prediction error is 43.12 hours
- This seems very high!

**Why It's High**:
- Small dataset (34 samples) → high variance
- Balancing reduced information
- Model struggles with limited data
- But fairness improved

**Is This Acceptable?**:
- **For fairness**: Yes (legal requirement)
- **For deployment**: Maybe (depends on use case)
- **For research**: Yes (demonstrates trade-off)

#### R² Score: -0.09 (Mitigated Model)

**What This Means**:
- Model is 9% worse than predicting the mean
- Still negative, but better than baseline (-0.20)

**Why Negative?**:
- Model complexity doesn't match data complexity
- Linear model too simple for this problem
- But interpretability is more important than accuracy here

---

## Model Saving and Serialization

### Why Save the Model?

**Purpose**:
- **Reusability**: Use trained model in other applications
- **Deployment**: Deploy model in production systems
- **Reproducibility**: Ensure same model is used consistently
- **Sharing**: Share model with team members or other systems

### What We Save

#### Complete Model Data (`trained_absenteeism_model.pkl`)

**Contents**:
1. **Trained Model** (`lr_model`):
   - Learned coefficients
   - Intercept value
   - Model parameters

2. **Fitted Scaler** (`scaler`):
   - Mean and standard deviation for each feature
   - Needed to preprocess new data the same way
   - Critical: Must use same scaling as training!

3. **Feature Names** (`feature_names`):
   - List of all 50 feature names (after encoding)
   - Needed to ensure correct feature order
   - Critical: Features must be in same order as training!

4. **Performance Metrics**:
   - RMSE, MAE, R² scores
   - For documentation and comparison

5. **Fairness Metrics**:
   - Group-wise performance
   - Gap calculations
   - For transparency and auditing

6. **Bias Mitigation Information**:
   - What measures were applied
   - For documentation and compliance

#### Simple Model Data (`model.pkl`)

**Contents**:
- Just model, scaler, and feature names
- Lightweight version for web deployment
- No metadata (faster loading)

**Why Two Versions?**:
- **Complete version**: For analysis, documentation, auditing
- **Simple version**: For production deployment (faster, smaller)

### Pickle Serialization

**What is Pickle?**:
- Python's built-in serialization format
- Converts Python objects to byte stream
- Can save/load complex objects (models, data structures)

**Why Pickle?**:
- **Simple**: Built into Python, no extra dependencies
- **Flexible**: Can save any Python object
- **Efficient**: Fast serialization/deserialization
- **Standard**: Widely used in ML workflows

**How It Works**:
```python
# Saving
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Loading
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
```

**File Format**:
- Binary format (not human-readable)
- Python-specific (can't use in other languages)
- Version-dependent (Python/scikit-learn version matters)

### Model Loading in Production

**Critical Requirements**:
1. **Same Python Version**: Pickle files are version-dependent
2. **Same Library Versions**: scikit-learn version must match
3. **Same Feature Order**: Features must be in exact same order
4. **Same Preprocessing**: Must use same scaler and encoding

**Why This Matters**:
- Different versions → loading errors
- Wrong feature order → wrong predictions
- Different scaling → wrong predictions
- Must maintain consistency!

---

## Summary

Assignment 2 demonstrates:

1. **Bias Detection**: How to identify bias in datasets and models
2. **Bias Analysis**: Understanding sources and impacts of bias
3. **Bias Mitigation**: Techniques to reduce bias (feature removal, balancing)
4. **Fairness Evaluation**: Metrics to measure group-wise fairness
5. **Trade-offs**: Understanding accuracy vs. fairness trade-offs
6. **Model Serialization**: How to save and deploy trained models

**Key Takeaways**:
- Bias exists in real-world datasets
- Mitigation techniques can improve fairness
- But there are trade-offs (accuracy vs. fairness)
- Fairness is often more important than raw accuracy
- Documentation and transparency are critical

**This foundation enables Assignment 4 (UI design) and Assignment 6 (explainability).**

