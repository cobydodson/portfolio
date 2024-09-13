#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 
# *Your name here*
# 
# ### About this template
# 
# This file is a **template** for filling out and submitting your final project. As such, I've created sub-sections along the lines of what we'd like to see. Your job is to **fill out** these sections, using the dataset and research question of your choice.
# 
# ### Things to be aware of
# 
# - Your project likely depends on **data**. Make sure any dataset you will analyze is *stored in your DataHub directory*, so you can submit it along with your project.  
# - Each of these sections will be assigned a point score. Make sure you add code cells in the relevant section, as needed.
# - The final project should be completed independently.

# ## Introduction (2 pts.)
# 
# Questions to answer:
# 
# 1. What dataset are you looking at? 
# 2. Where/how was it created? 
# 3. What research question(s) will you be asking? 
# 
# These should be answered in Markdown.

# I am looking at the dataset that surveyed student alcohol consumption. 
# This dataset was created by a survey of students taking math and portugese language courses/ 
# The research question i am asking is "Does alcohol consumption affect academic performance in college students?" This will be meaasured by both study time and numer of past class failures. 

# ## Data (3 pts.)
# 
# This section should contain **descriptive statistics** about your data. This includes (but is not limited to):
# 
# 1. Overall `shape` of the data. 
# 2. Summary statistics, e.g., central tendency, variability, of key **features** (i.e., columns).
# 3. Histograms / count-plots of key features (i.e., columns). 
# 4. Information about missing values, if relevant.  
# 5. Information about **merging** datasets, if relevant.
# 
# These should be **answered** using Python code (but can be written in Markdown if you prefer).

# In[1]:


# YOUR CODE HERE
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss
import statsmodels.formula.api as smf

from sklearn.metrics import confusion_matrix


# In[2]:


#Overall shape of the data
df_mat = pd.read_csv("student-mat.csv")
df_por = pd.read_csv("student-por.csv")

df_student = pd.merge(df_mat, df_por, on=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet", "Dalc", "Walc", "studytime", "failures"])
print(df_student.shape)


# In[3]:


#Central tendency alcohol consumption over a week
df_student['weekalc'] = df_student['Dalc'] + df_student['Walc']
mean_alc = df_student['weekalc'].mean()
median_alc = df_student['weekalc'].median()
std_alc = df_student['weekalc'].std()
print(mean_alc)
print(median_alc)
print(std_alc)


# In[4]:


#Central tendency weekly study time
mean_study = df_student['studytime'].mean()
median_study = df_student['studytime'].median()
std_study = df_student['studytime'].std()
print(mean_study)
print(median_study)
print(std_study)


# In[5]:


#Central tendency past failures
mean_failure = df_student['failures'].mean()
median_failure = df_student['failures'].median()
std_failure = df_student['failures'].std()
print(mean_failure)
print(median_failure)
print(std_failure)


# In[6]:


#Histogram of alcohol consumption
p_alc = plt.hist(df_student['weekalc'], bins = 5)
p_alc


# In[7]:


#Histogram of study time
p_study = plt.hist(df_student['studytime'], bins = 5)
p_study


# In[8]:


#Histogram of failures
p_fail = plt.hist(df_student['failures'], bins = 5)
p_fail


# In[ ]:





# ## Visualizations (4 pts.)
# 
# This section should contain:
# 
# - **2-3 graphs** showing specific patterns or features you'd like to highlight. 
# - Each visualization should be accompanied by a **short (1-2 sentences) description** of what you think it shows.
# 
# These should be **produced** using Python code (but the descriptions can be written in Markdown if you prefer).

# In[9]:


summary = df_student[['weekalc', 'studytime']].groupby("weekalc").mean().reset_index()
summary


# In[10]:


summary['weekalc'] = summary['weekalc'].apply(lambda x: str(x))
summary['weekalc']


# This bar graph represents the weekly alcohol conumption based on the hours weekly study time. 

# In[35]:


plt.bar(x = summary['weekalc'],
       height = summary['studytime'])
plt.xlabel('Weekly Alcohol Consumption')
plt.ylabel('Studytime')
plt.title("Weekly Alcohol Consumption vs Studytime")


# This seaborn bar plot represents the amount of weekly alcohol consumption when compared to failures.

# In[27]:


sns.barplot(data = df_student, x = 'failures', 
               y = 'weekalc')


# This bar plot represents the relationship between study time and weekly alcohol consumption. It also shows the differences
# #between students who have had a failure in a past class and who has not.

# In[51]:


df_student['Past Class Failure'] = (df_student['failures'] > 0)
sns.barplot(data = df_student, x = "weekalc",
           y = "studytime", hue = "Past Class Failure")


# ## Analyses (4 pts.)
# 
# This section should contain:
# 
# - **2-3 analyses** using methods discussed in class (e.g., linear regression, logistic regression, etc.) to address your question.
# - Each analysis should be accompanied by a short (1-3 sentences) **interpretation**. 
# - Should also include **evaluation** of your model somehow, e.g., $R^2$, AIC, etc. 
# 
# These should be **produced** using Python code (but the interpretations can be written in Markdown if you prefer).

# I created a linear regression model that looks at the correlation between the weekly alcohol consumption and study time. The model showed a negative correlation of -0.514773 between alcohol consumption and study time, but the r squared value was 0.0519, suggesting that there are other factors that may contribute. 

# In[58]:


mod_alc = smf.ols(data = df_student, formula = "weekalc ~ studytime").fit()
print(mod_alc.params)
print(mod_alc.rsquared)


# I used three logistic models to represent the relationship that consuming over three drinks has on study time, as well as previous class failures. The models use Akaike Information Crierion to measure the accuracy and complexity of the models. The AIC showed that study time provides a better fit for measuring alcohol consumption.

# In[77]:


df_student['alc_three'] = (df_student['weekalc'] > 3).astype(int)
mod_study = smf.logit(formula = "alc_three ~ studytime", data = df_student).fit()
mod_study.params
aic_study = mod_study.aic
aic_study


# In[79]:


mod_fail = smf.logit(formula = "alc_three ~ failures", data = df_student).fit()
aic_fail = mod_fail.aic
aic_fail


# In[82]:


mod_both = smf.logit(formula = "alc_three ~ failures + studytime", data = df_student).fit()
aic_both = mod_both.aic
aic_both


# ## Limitations and Ethical Issues (3 pts.)
# 
# This section should contain a discussion of any **limitations** to your analysis, as well as any **ethical issues**, if relevant.
# 
# - Limitations could range from issues in the data (e.g., poor generalizability, biased sample) to the assumptions of the analysis (e.g., homoscedasticity vs. heteroscedasticity), and so on.
# - Ethical issues should focus on concepts covered in class, e.g., relating to bias and/or privacy.  
# 
# These should be answered in Markdown.

# One limitation of this study could be poor generalizability, as this is a small sample size that only surveyed two schools. If a study is looking at college students as a whole population, it may not be correctly represented. Another issue may be recall bias, which occurs when participants are surveryed. This is when participants incorrectly recall information, either on accident or if they believe that one outcome is better than another. In this scenario, a student may report more study time or less alcohol consumption. 
# 
# When it comes to ethics, there are some concerns of privacy due to de-anonymization, but it isnt much of an issue. This dataset uses surveyed data from anonymous college students. This means that by filling this survey out they have consented to their data being used, only anonymously. Another issue that this can provide is selection bias, as only college students that were interested in the survey filled it out. Based on the dataset used, there is little concern for ethics if any. 

# ## Conclusion (1 pt.)
# 
# Draw a conclusion about the dataset and the questions you posed.
# 
# These should be answered in Markdown.

# There is a correlation between weekly student alcohol consumption and a decrease in academic performance, mainly when considering study time. Although there is a correlation, it was not as significant as expected and there are other factors that were not explored and could make a difference in the results. 
