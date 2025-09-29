# %% [markdown]
# # Final edits before sharing

# %%
import pandas as pd
# Import dataframe
syn_df= pd.read_csv("syn_df_full_deduped.csv", encoding = "latin-1")
syn_df.info()


# %%
print(syn_df.head(1)) 
print(syn_df.columns.to_list())
print(syn_df['age_grp'].unique())
print(syn_df.groupby('age_grp').count().reset_index())


# %%
# Create age column
import numpy as np
np.random.seed(123)
# Create age column based on age_grp
syn_df['age'] = syn_df['age_grp']  # initialize or create the column

for i in range(len(syn_df)):
    age_grp = syn_df.loc[i, 'age_grp']
    
    if age_grp == '<19':
        syn_df.loc[i, 'age'] = np.random.randint(16, 20)
    elif age_grp == '20???24':
        syn_df.loc[i, 'age'] = np.random.randint(20, 25)
    elif age_grp == '25???29':
        syn_df.loc[i, 'age'] = np.random.randint(25, 30)
    elif age_grp == '30???34':
        syn_df.loc[i, 'age'] = np.random.randint(30, 35)
    elif age_grp == '35???39':
        syn_df.loc[i, 'age'] = np.random.randint(35, 40)
    elif age_grp == '40???44':
        syn_df.loc[i, 'age'] = np.random.randint(40, 45)
    elif age_grp == '45???49':
        syn_df.loc[i, 'age'] = np.random.randint(45, 50)
    elif age_grp == '50???54':
        syn_df.loc[i, 'age'] = np.random.randint(50, 55)
    elif age_grp == '55+':
        syn_df.loc[i, 'age'] = 55

syn_df['age'].describe()


# %%
# Drop age_grp column and synth_id column
syn_df = syn_df.drop('age_grp', axis=1)
syn_df = syn_df.drop('synth_id', axis = 1)
syn_df_rep = syn_df.copy()

# %%
# Replace CODEASBLANK with random characters for columns city	country	gender	educstat	digitools	industry	careerstg

random_choices_blanks = ["na", "_", " ", "/", ".", ". ", " _", "-", "not applicable", "None", "none", "NONE"]
replace_cols = ['city', 'country', 'gender', 'educstat', 'digitools', 'industry', 'careerstg', 'salary']

def replace_codeasblank(val, col):
    if val == 'CODEASBLANK':
        if col in replace_cols:
            return ""
        else:
            return np.random.choice(random_choices_blanks)
    return val

# Apply replacement across the entire DataFrame
syn_df_rep = syn_df_rep.apply(lambda col: [replace_codeasblank(val, col.name) for val in col])


# %%
print(syn_df_rep.head())

# %%
print(syn_df_rep['salary'].unique())

# %%
# Retain salary if educstat is Secondary school student and salary is '15,000 and below' or '25,001 to 35,000' or '35,001 to 45,000' or '75,001 to 85,000', else replace with ""
valid_salaries = ["15,000 and below", "25,001 to 35,000", "35,001 to 45,000", "75,001 to 85,000"]
condition1 = syn_df_rep['educstat'] == "Secondary school student"
condition2 = ~syn_df_rep['salary'].isin(valid_salaries)
syn_df_rep['salary'] = np.where(condition1 & condition2, "", syn_df_rep['salary'])

# %%
# Check if high school salaries are correct
educ_pivot = syn_df_rep.pivot_table(index='educstat', columns='salary', aggfunc='size',fill_value=0).reset_index()
educ_pivot

# %%
# Pivot table for salary
salary_by_educ = syn_df_rep.pivot_table(index='salary', columns='educstat', aggfunc='size',fill_value=0)
pd.DataFrame(salary_by_educ)
salary_by_educ.drop('', axis = 1, inplace = True)
salary_by_educ.drop('', axis = 0, inplace = True)

salary_by_educ

# %%
# Import matplotlib and seaborn 

import matplotlib.pyplot as plt
import seaborn as sns 

# %%
# Plot salary by educstat to see if the Secondary school student salaries reflect actual data

salary_by_educ.plot(kind = "bar", stacked = True, figsize = (10, 6))
plt.xlabel('Salary Level')
plt.ylabel('Count')
plt.title('Education Status Distribution by Salary Level')
plt.legend(title='Education Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Check career stage vs. salary and student in career stage
career_salary = syn_df_rep.pivot_table(index='careerstg', columns='salary', aggfunc='size',fill_value=0).reset_index()
student_career = career_salary[career_salary['careerstg'] == 'Student/ New grad/ Career Break']
student_career

# %%
# Zero out the salary of student in career stage
syn_df_rep['salary'] = np.where(syn_df_rep['careerstg'] == 'Student/ New grad/ Career Break', "", syn_df_rep['salary']) 
career_salary = syn_df_rep.pivot_table(index='careerstg', columns='salary', aggfunc='size',fill_value=0).reset_index()
student_career = career_salary[career_salary['careerstg'] == 'Student/ New grad/ Career Break']
student_career

# %%
syn_df_rep.to_csv("syn_df_rep.csv", index = False)

# %%
syn_df_rep.info()

# %%
# Rename cols to actual survey questions
syn_df_for_sharing = syn_df_rep.copy()
dfheaders = pd.read_csv("columns.csv", header=None)
var_names = dfheaders.iloc[0]
survey_questions = dfheaders.iloc[1]
df_dict = dict(zip(var_names, survey_questions))
syn_df_for_sharing.rename(columns=df_dict, inplace=True)


# %%
print(syn_df_for_sharing.head())

# %%
syn_df_for_sharing.to_csv("syn_df_for_sharing.csv", index = False)


