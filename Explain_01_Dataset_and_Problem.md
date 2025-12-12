# Explain 01: Dataset and Problem Statement
## Deep Dive into the Absenteeism Prediction Project

---

## Table of Contents
1. [Why This Project Exists](#why-this-project-exists)
2. [The Real-World Problem](#the-real-world-problem)
3. [Dataset Overview](#dataset-overview)
4. [Dataset Source and Context](#dataset-source-and-context)
5. [Complete Feature Breakdown](#complete-feature-breakdown)
6. [Target Variable Explanation](#target-variable-explanation)
7. [Data Quality and Characteristics](#data-quality-and-characteristics)
8. [Why Machine Learning is Appropriate](#why-machine-learning-is-appropriate)

---

## Why This Project Exists

### The Academic Context
This project is part of **CS698Y - Human-AI Interaction** course at IIT Kanpur. The course focuses on:
- Understanding how humans interact with AI systems
- Building responsible AI systems that are fair, transparent, and trustworthy
- Creating interfaces that help users understand and trust AI predictions
- Evaluating AI systems using frameworks like Microsoft's HAX (Human-AI eXperience) principles

### The Learning Objectives
1. **Assignment 2**: Learn to identify and mitigate bias in machine learning models
2. **Assignment 4**: Design user interfaces that communicate model capabilities transparently
3. **Assignment 6**: Implement explainability features to help users understand predictions

### Why Absenteeism Prediction?
Absenteeism prediction is an ideal problem for this course because:
- **Real-world relevance**: HR departments need to predict and manage employee absenteeism
- **Ethical complexity**: Predictions can affect employee evaluations, promotions, and job security
- **Bias potential**: Models can unfairly discriminate based on age, education, or other protected attributes
- **Interpretability need**: HR professionals need to understand why predictions are made
- **Fairness critical**: Unfair predictions can lead to discrimination lawsuits and ethical violations

---

## The Real-World Problem

### What is Employee Absenteeism?
Employee absenteeism refers to the **habitual non-presence of an employee at their job**. It's measured in hours or days that an employee is absent from work during a specific period.

### Why is Absenteeism a Problem?
1. **Financial Impact**:
   - Direct costs: Paid sick leave, overtime for replacement workers
   - Indirect costs: Reduced productivity, project delays, customer dissatisfaction
   - Estimated cost: $225.8 billion annually in the US alone

2. **Operational Impact**:
   - Disrupted workflows and project timelines
   - Increased workload on present employees
   - Reduced team morale and cohesion
   - Difficulty in resource planning and scheduling

3. **Strategic Impact**:
   - Inability to meet business objectives
   - Reduced competitive advantage
   - Poor customer service quality
   - Difficulty in long-term planning

### Why Predict Absenteeism?
Predicting absenteeism allows organizations to:
- **Proactive Management**: Identify employees at risk of high absenteeism before it becomes a problem
- **Resource Planning**: Allocate resources and plan schedules more effectively
- **Intervention Programs**: Develop targeted wellness or support programs for at-risk employees
- **Cost Reduction**: Minimize the financial impact of unexpected absences
- **Fair Evaluation**: Distinguish between legitimate health issues and other causes

### The Ethical Challenge
However, predicting absenteeism raises serious ethical concerns:
- **Privacy**: Monitoring employee behavior and health information
- **Discrimination**: Models might unfairly penalize certain demographic groups
- **Stigma**: Employees with legitimate health issues might be labeled as "problem employees"
- **Bias**: Historical data might reflect past discrimination, perpetuating it in predictions

**This is why our project focuses on fairness, transparency, and explainability.**

---

## Dataset Overview

### Basic Statistics
- **Total Records**: 740 observations
- **Features**: 21 attributes (20 input features + 1 target variable)
- **Employees**: 36 unique employees
- **Time Period**: Multiple time periods (employees appear multiple times)
- **Data Format**: CSV file with semicolon (;) delimiter
- **Missing Values**: None (complete dataset)
- **Duplicates**: 34 duplicate records identified

### Dataset Structure
The dataset represents **multiple observations per employee** over time. This means:
- Each row is a specific absence event or time period
- The same employee (ID) appears in multiple rows
- This allows us to track patterns over time
- We can identify employees with recurring absence patterns

### Why This Structure Matters
- **Temporal Patterns**: We can identify seasonal patterns, day-of-week effects
- **Individual Patterns**: Some employees may have consistent absence patterns
- **Contextual Factors**: Each absence event has specific contextual information (month, day, reason)

---

## Dataset Source and Context

### Origin
The dataset comes from the **UCI Machine Learning Repository** (University of California, Irvine), which is a well-known source for machine learning datasets used in research and education.

### Original Purpose
The dataset was collected to study:
- Factors influencing employee absenteeism
- Patterns in workplace absence
- Relationship between employee characteristics and absenteeism

### Industry Context
While the exact industry isn't specified, the dataset characteristics suggest:
- **Manufacturing or Service Industry**: Based on work load metrics and shift patterns
- **Structured Work Environment**: Clear work schedules, target-based performance
- **Health-Conscious Organization**: Tracks health-related absence reasons in detail

### Data Collection Method
The dataset appears to be collected through:
- **HR Records**: Official absence records from HR systems
- **Employee Surveys**: Self-reported information (social habits, family status)
- **Workplace Metrics**: Objective measurements (work load, transportation expense)

---

## Complete Feature Breakdown

### Feature Categories
The 21 features can be organized into 5 main categories:

1. **Demographic Features** (3 features)
2. **Workplace Features** (4 features)
3. **Behavioral Features** (3 features)
4. **Health Features** (3 features)
5. **Temporal Features** (4 features)
6. **Target Variable** (1 feature)

---

### 1. Demographic Features

#### **ID** (Employee Identifier)
- **Type**: Integer (1-36)
- **Values**: Unique identifier for each employee
- **Why It Exists**: To track individual employees across multiple records
- **Why We Remove It**: 
  - Not a predictive feature (just an identifier)
  - Could lead to overfitting (model memorizing specific employees)
  - Privacy concern (identifying specific individuals)
  - Not generalizable (can't use for new employees)

#### **Age** (Employee Age)
- **Type**: Integer
- **Range**: Typically 20-60 years
- **Why It Matters**: 
  - Older employees may have more health issues
  - Younger employees may have different absence patterns (education, family)
  - Age-related health conditions increase with age
- **Bias Concern**: 
  - Age is a protected attribute in many jurisdictions
  - Age discrimination is illegal
  - Models must not unfairly penalize older or younger workers
- **How We Handle It**: 
  - We group ages into categories (18-30, 31-40, 41-50, 50+)
  - We monitor fairness across age groups
  - We balance representation in training data

#### **Education** (Education Level)
- **Type**: Integer (1-4)
- **Values**:
  - 1: High School
  - 2: Graduate
  - 3: Post Graduate
  - 4: Master and Doctor
- **Why It Matters**:
  - Education level may correlate with job type and stress levels
  - Higher education might mean more responsibility and stress
  - Education might affect health awareness and preventive care
- **Bias Concern**:
  - Education can be a proxy for socioeconomic status
  - May correlate with other protected attributes
  - Should not be used to discriminate
- **Distribution Problem**:
  - Level 1: 611 records (82.6%) - **Severely over-represented**
  - Level 2: 46 records (6.2%) - Under-represented
  - Level 3: 79 records (10.7%) - Under-represented
  - Level 4: 4 records (0.5%) - **Extremely under-represented**
- **How We Handle It**: 
  - We balance education levels in training data
  - We monitor fairness metrics across education groups

#### **Service Time** (Years of Service)
- **Type**: Integer
- **Range**: Typically 1-30 years
- **Why It Matters**:
  - Longer service might mean more job satisfaction (less absence)
  - Or more burnout (more absence)
  - Seniority might affect absence patterns
- **How We Use It**: 
  - We group into categories (0-5, 6-10, 11-15, 15+ years)
  - We monitor fairness across service time groups

---

### 2. Workplace Features

#### **Transportation Expense** (Cost of Commuting)
- **Type**: Integer (currency units)
- **Range**: Typically 100-400
- **Why It Matters**:
  - Higher transportation costs might discourage attendance
  - Financial stress from commuting might affect health
  - Distance and cost are related
- **Real-World Context**:
  - Employees with expensive commutes might be more likely to miss work
  - Financial burden can cause stress-related absences
  - Weather or transportation issues affect attendance

#### **Distance from Residence to Work** (Kilometers)
- **Type**: Integer
- **Range**: Typically 1-50 km
- **Why It Matters**:
  - Longer commutes increase likelihood of absence
  - Transportation delays more likely with longer distances
  - Fatigue from long commutes affects health
- **Relationship with Transportation Expense**:
  - Generally correlated (longer distance = higher cost)
  - But not always (public transport vs. car)
- **Practical Impact**:
  - Employees far from work might miss more days
  - Weather, traffic, or transportation issues more impactful

#### **Work Load Average/day** (Work Units per Day)
- **Type**: Float
- **Range**: Typically 200-350 units
- **What It Represents**: 
  - Amount of work assigned per day
  - Could be tasks, units produced, or work hours
  - Higher values = more work pressure
- **Why It Matters**:
  - High work load → stress → health issues → absences
  - Burnout from excessive work load
  - Work-life balance affects attendance
- **Measurement Note**: 
  - This is NOT hours worked
  - It's a work unit metric (tasks, units, etc.)
  - Higher values indicate more work pressure

#### **Hit Target** (Performance Target Achievement)
- **Type**: Integer (0-100, or binary 0/1)
- **Values**: 
  - 0: Did not hit target
  - 1: Hit target
  - Or percentage (0-100)
- **Why It Matters**:
  - Employees missing targets might have stress-related absences
  - Performance pressure affects health
  - Target achievement indicates job satisfaction
- **Interpretation**:
  - Low target achievement → stress → more absences
  - High target achievement → satisfaction → fewer absences

---

### 3. Behavioral Features

#### **Social Drinker** (Alcohol Consumption)
- **Type**: Binary (0 or 1)
- **Values**:
  - 0: Does not drink socially
  - 1: Drinks socially
- **Why It Matters**:
  - Alcohol consumption can affect health
  - Hangovers might cause absences
  - Social drinking might indicate lifestyle factors
- **Ethical Consideration**:
  - This is personal lifestyle information
  - Should not be used to discriminate
  - Privacy concern
- **How We Use It**: 
  - As a potential health indicator
  - Not as a basis for discrimination

#### **Social Smoker** (Tobacco Use)
- **Type**: Binary (0 or 1)
- **Values**:
  - 0: Does not smoke
  - 1: Smokes socially
- **Why It Matters**:
  - Smoking affects health and can cause absences
  - Respiratory issues from smoking
  - Health-related lifestyle factor
- **Distribution**: 
  - Most employees don't smoke (686 non-smokers vs. 54 smokers)
  - Highly imbalanced feature
- **Ethical Consideration**:
  - Personal health information
  - Should not be used to penalize employees

#### **Pet** (Pet Ownership)
- **Type**: Integer (0-8, or binary)
- **Values**: 
  - 0: No pets
  - 1-8: Number of pets owned
- **Why It Matters**:
  - Pet care responsibilities might affect attendance
  - Pet-related emergencies might cause absences
  - Pet ownership might indicate lifestyle and responsibility
- **Interpretation**:
  - Pet owners might have different absence patterns
  - Pet emergencies are legitimate absence reasons
  - Lifestyle indicator

---

### 4. Health Features (Removed for Bias Mitigation)

#### **Weight** (Body Weight in kg)
- **Type**: Integer
- **Range**: Typically 50-120 kg
- **Why We Remove It**:
  - **Proxy for Gender**: Weight distributions differ by gender
  - **Proxy for Age**: Weight changes with age
  - **Health Discrimination**: Using weight to predict absence is discriminatory
  - **Privacy**: Personal health information
  - **Not Job-Related**: Weight doesn't directly affect job performance

#### **Height** (Body Height in cm)
- **Type**: Integer
- **Range**: Typically 150-200 cm
- **Why We Remove It**:
  - **Proxy for Gender**: Height distributions differ by gender
  - **Proxy for Age**: Height can indicate age (children vs. adults)
  - **Not Predictive**: Height doesn't directly affect absenteeism
  - **Privacy**: Personal physical information

#### **Body Mass Index (BMI)**
- **Type**: Integer
- **Calculation**: Weight (kg) / Height (m)²
- **Why We Remove It**:
  - **Derived Feature**: Calculated from Weight and Height (already removed)
  - **Health Discrimination**: Using BMI to predict absence is discriminatory
  - **Proxy Attribute**: Encodes gender, age, and health information
  - **Not Job-Related**: BMI doesn't directly affect job performance

**This is a critical bias mitigation step - removing features that encode protected attributes.**

---

### 5. Temporal Features

#### **Month of Absence** (Calendar Month)
- **Type**: Integer (1-12)
- **Values**: 
  - 1: January
  - 2: February
  - ... 12: December
- **Why It Matters**:
  - **Seasonal Patterns**: 
    - Winter months (Dec-Feb) → more illness (flu season)
    - Summer months (Jun-Aug) → vacation time, family activities
    - Holiday seasons → more absences
  - **Weather Effects**: 
    - Cold weather → more illness
    - Extreme weather → transportation issues
  - **Business Cycles**: 
    - End of year → more absences (holidays, year-end fatigue)
    - Beginning of year → fresh start, fewer absences
- **Distribution**: 
  - Month 3 (March): 87 records (highest)
  - Month 2 (February): 72 records
  - Month 10 (October): 71 records
- **Practical Use**: 
  - HR can plan for seasonal patterns
  - Allocate resources based on expected absences

#### **Day of the Week** (Weekday)
- **Type**: Integer (2-6)
- **Values**:
  - 2: Monday
  - 3: Tuesday
  - 4: Wednesday
  - 5: Thursday
  - 6: Friday
- **Why It Matters**:
  - **Monday Effect**: More absences on Mondays (weekend recovery, reluctance to start week)
  - **Friday Effect**: More absences on Fridays (extended weekends)
  - **Mid-Week**: Fewer absences (Tuesday-Thursday)
  - **Pattern Recognition**: Employees might have consistent day-of-week patterns
- **Distribution**:
  - Monday (2): 161 records (highest)
  - Friday (6): 125 records
  - Mid-week: More balanced
- **Practical Use**: 
  - Schedule important meetings mid-week
  - Plan for Monday/Friday absences

#### **Seasons** (Time of Year)
- **Type**: Integer (1-4)
- **Values**:
  - 1: Spring (March-May)
  - 2: Summer (June-August)
  - 3: Fall (September-November)
  - 4: Winter (December-February)
- **Why It Matters**:
  - **Health Patterns**: 
    - Winter → more illness (cold, flu)
    - Spring → allergies, seasonal illnesses
    - Summer → vacation time, heat-related issues
    - Fall → back-to-school, family responsibilities
  - **Lifestyle Patterns**: 
    - Summer → vacations, family time
    - Winter → holidays, family gatherings
- **Distribution**: 
  - Winter (4): 195 records (highest - illness season)
  - Summer (2): 192 records (vacation season)
  - Spring (1): 170 records
  - Fall (3): 183 records
- **Relationship with Month**: 
  - Seasons are derived from months
  - Provides higher-level pattern recognition
  - Less granular than month but captures seasonal trends

#### **Reason for Absence** (ICD-10 Classification)
- **Type**: Integer (0-28)
- **Values**: 
  - 0: No absence (present at work)
  - 1-21: Various disease categories (ICD-10 codes)
  - 22-28: Other reasons (follow-up, consultation, etc.)
- **ICD-10 Categories**:
  - 1: Certain infectious and parasitic diseases
  - 2: Neoplasms (cancers)
  - 3: Diseases of the blood
  - 4: Endocrine, nutritional and metabolic diseases
  - 5: Mental and behavioural disorders
  - 6: Diseases of the nervous system
  - 7: Diseases of the eye and adnexa
  - 8: Diseases of the ear and mastoid process
  - 9: Diseases of the circulatory system
  - 10: Diseases of the respiratory system
  - 11: Diseases of the digestive system
  - 12: Diseases of the skin and subcutaneous tissue
  - 13: Diseases of the musculoskeletal system and connective tissue
  - 14: Diseases of the genitourinary system
  - 15: Pregnancy, childbirth and the puerperium
  - 16: Certain conditions originating in the perinatal period
  - 17: Congenital malformations, deformations and chromosomal abnormalities
  - 18: Abnormal clinical and laboratory findings
  - 19: Injury, poisoning and certain other consequences of external causes
  - 20: External causes of morbidity and mortality
  - 21: Factors influencing health status and contact with health services
  - 22: Patient follow-up
  - 23: Medical consultation
  - 24: Blood donation
  - 25: Laboratory examination
  - 26: Unjustified absence
  - 27: Physiotherapy
  - 28: Dental consultation
- **Why It Matters**:
  - **Health Patterns**: Different reasons indicate different health issues
  - **Predictive Value**: Some reasons might predict future absences
  - **Intervention Opportunities**: Different reasons require different interventions
- **Distribution**:
  - Reason 23 (Medical consultation): 149 records (highest)
  - Reason 28 (Dental consultation): 112 records
  - Reason 27 (Physiotherapy): 69 records
  - Reason 0 (No absence): 43 records
- **Ethical Consideration**:
  - Health information is sensitive
  - Should not be used to discriminate
  - Privacy concerns

---

### 6. Family and Personal Features

#### **Son** (Number of Children)
- **Type**: Integer (0-4+)
- **Values**: Number of children/dependents
- **Why It Matters**:
  - **Family Responsibilities**: 
    - Children's illnesses → parent absences
    - School events → parent absences
    - Childcare issues → absences
  - **Work-Life Balance**: 
    - More children → more family responsibilities
    - Stress from family obligations
  - **Life Stage Indicator**: 
    - Younger employees with children might have different patterns
- **Distribution**:
  - 0 children: 298 records (40.3%)
  - 1 child: 229 records (30.9%)
  - 2 children: 156 records (21.1%)
  - 3+ children: 57 records (7.7%)
- **Ethical Consideration**:
  - Family status is personal information
  - Should not be used to discriminate against parents
  - Gender bias concern (mothers vs. fathers)

#### **Disciplinary Failure** (Disciplinary Action)
- **Type**: Binary (0 or 1)
- **Values**:
  - 0: No disciplinary action
  - 1: Has disciplinary action
- **Why It Matters**:
  - **Behavioral Indicator**: 
    - Employees with disciplinary issues might have more absences
    - Or absences might lead to disciplinary action
  - **Job Satisfaction**: 
    - Disciplinary issues might indicate job dissatisfaction
    - Stress from disciplinary action
- **Distribution**:
  - No disciplinary action: 700 records (94.6%)
  - Has disciplinary action: 40 records (5.4%)
  - **Highly imbalanced** - most employees have no disciplinary issues
- **Causality Question**:
  - Does absence cause disciplinary action?
  - Or does disciplinary action cause more absences?
  - Correlation doesn't imply causation

---

## Target Variable Explanation

### **Absenteeism time in hours**
- **Type**: Integer
- **Range**: 0-120 hours
- **Mean**: 6.92 hours
- **Median**: 3.00 hours
- **Standard Deviation**: 13.33 hours

### What Does This Represent?
This is a **regression target** (continuous variable) representing:
- **Total hours absent** during a specific time period
- **Not just binary** (absent/present), but **how much** time was missed
- Allows for **partial absences** (half-days, few hours)

### Why Hours Instead of Days?
- **More Granular**: Captures partial absences
- **More Accurate**: Some absences are just a few hours
- **Better for Planning**: HR can plan resources more precisely
- **More Informative**: Distinguishes between 1 hour and 8 hours of absence

### Distribution Characteristics
- **Right-Skewed**: Most absences are small (0-10 hours), few are very large (50+ hours)
- **Zero-Inflated**: Many records have 0 hours (no absence)
- **Outliers**: Some extreme values (up to 120 hours) - likely extended medical leave

### Interpretation Examples
- **0 hours**: Employee was present (no absence)
- **2-4 hours**: Short absence (half-day, doctor appointment)
- **8 hours**: Full day absence
- **16-24 hours**: Multiple days absence
- **50+ hours**: Extended absence (medical leave, vacation)

### Why This is a Regression Problem
- **Continuous Output**: We predict a number (hours), not a category
- **Magnitude Matters**: 2 hours vs. 20 hours is very different
- **Linear Regression Appropriate**: 
  - Simple and interpretable
  - Coefficients show feature importance
  - Good baseline for comparison

---

## Data Quality and Characteristics

### Data Completeness
- **No Missing Values**: All 740 records have complete data
- **Why This Matters**: 
  - No need for imputation
  - No data loss from missing value handling
  - Clean dataset for analysis

### Data Consistency
- **34 Duplicate Records**: Some records appear multiple times
- **Why Duplicates Exist**:
  - Same employee, same time period, recorded twice
  - Data entry errors
  - Multiple absence events in same period
- **How We Handle It**: 
  - We keep duplicates for now (they might represent multiple events)
  - We monitor their impact on model performance

### Data Balance Issues

#### Age Distribution
- **18-30 years**: 177 samples (23.9%) - Balanced
- **31-40 years**: 422 samples (57.0%) - **Over-represented**
- **41-50 years**: 132 samples (17.8%) - Balanced
- **50+ years**: 9 samples (1.2%) - **Severely under-represented**

**Problem**: Model will be biased toward 31-40 age group predictions.

#### Education Distribution
- **Level 1**: 611 records (82.6%) - **Severely over-represented**
- **Level 2**: 46 records (6.2%) - Under-represented
- **Level 3**: 79 records (10.7%) - Under-represented
- **Level 4**: 4 records (0.5%) - **Extremely under-represented**

**Problem**: Model will be biased toward education level 1 predictions.

#### Target Variable Distribution
- **Mean**: 6.92 hours
- **Median**: 3.00 hours (much lower than mean)
- **Right-Skewed**: Most values are small, few are very large
- **Zero-Inflated**: Many zero values (no absence)

**Problem**: Model might struggle with rare high-absence cases.

---

## Why Machine Learning is Appropriate

### Why Not Simple Rules?
Simple rules like "if age > 50, predict high absence" don't work because:
- **Multiple Factors**: Absenteeism depends on many factors simultaneously
- **Non-Linear Relationships**: Factors interact in complex ways
- **Context-Dependent**: Same factor might have different effects in different contexts
- **Pattern Recognition**: ML can find patterns humans might miss

### Why Linear Regression?
We chose Linear Regression because:
1. **Interpretability**: 
   - Coefficients show how each feature affects prediction
   - Easy to explain to non-technical users
   - Transparent decision-making process

2. **Baseline Performance**:
   - Establishes a baseline for comparison
   - Simple model to understand before adding complexity
   - Good starting point for fairness analysis

3. **Computational Efficiency**:
   - Fast training and prediction
   - No hyperparameter tuning needed
   - Works well with small datasets

4. **Fairness Analysis**:
   - Easy to analyze coefficients for bias
   - Can identify which features contribute to unfairness
   - Simple to implement fairness constraints

### Limitations of Linear Regression
- **Assumes Linearity**: Real relationships might be non-linear
- **Limited Complexity**: Can't capture complex interactions
- **Sensitive to Outliers**: Extreme values can skew predictions
- **Feature Engineering Required**: Need to encode categorical variables properly

### Why Not More Complex Models?
We could use Random Forest, Neural Networks, or XGBoost, but:
- **Less Interpretable**: Harder to explain predictions
- **More Complex**: Harder to analyze for bias
- **Overfitting Risk**: Small dataset (740 records) might lead to overfitting
- **Fairness Analysis**: More difficult to understand bias sources

**For this project, interpretability and fairness analysis are more important than raw accuracy.**

---

## Summary

This dataset represents a **real-world HR problem** with **serious ethical implications**. The features capture:
- **Demographic information** (age, education) - potential bias sources
- **Workplace factors** (work load, distance) - legitimate predictors
- **Health information** (weight, height, BMI) - removed for bias mitigation
- **Temporal patterns** (month, day, season) - useful for planning
- **Behavioral factors** (drinking, smoking) - lifestyle indicators

The target variable (absenteeism hours) is a **continuous regression target** that allows for:
- **Granular predictions** (hours, not just days)
- **Partial absences** (half-days, few hours)
- **Better resource planning** (precise time estimates)

The dataset has **significant imbalance issues** that require:
- **Bias mitigation techniques** (balancing, feature removal)
- **Fairness monitoring** (group-wise performance metrics)
- **Transparent communication** (users need to understand limitations)

**This foundation sets the stage for Assignment 2 (bias analysis), Assignment 4 (UI design), and Assignment 6 (explainability).**

