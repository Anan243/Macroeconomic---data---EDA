# Macroeconomic---data---EDA
import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\BALAJI\\Downloads\\global-econ-analysis\\Global Economy Indicators.csv")


print("Overview of what dataset contains :- ")
print("-----head------","\n",df.head(10),"\n")
print("-----tail-----","\n",df.tail(),"\n")
print("-----shape-----","\n",df.shape,"\n")
print("-----Column Names-----","\n",df.columns,"\n")
print("-----Information-----","\n")
df.info()
print("table summary Statistic:-","\n",df.describe(),"\n")


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Strip out anything in parentheses to make column names shorter
df.columns = df.columns.str.split('(').str[0].str.strip().str.lower().str.replace(" ", "_")



'''This code is used to standardize messy column names by removing accidental edge spaces, 
eliminating tricky uppercase letters,
 and replacing middle spaces with underscores.
Doing this prevents annoying "column not found" errors and makes your columns much
 easier and faster to type while writing code. Streamlines for ease of use when calling
 column names for code. Do not have to remember details like capital or small or spacing.'''

column_list = df.columns.tolist()
print("---- Modified and Streamlined column names listed below ------")
n = 0
for i in column_list:
    n = n+1
    print(n,". ",i)



print("----- Missing Values----- ","\n")
t = 0  
for i in df.columns:
    t = t + 1    
    # 2. Target the specific column first, then apply .isnull().sum() this is given in
        #code below df[i] is for that specific ith column
    s = df[i].isnull().sum()    
    print(t, ". ", i, " -> ", s)



# Now recalculate and print
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.sort_values(ascending=False))

print("----- Duplicated Rows----- ","\n")
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)
print("Duplicate Count:", duplicate_rows.shape[0])
df = df.drop_duplicates()


print("----- Numeric columns----- ","\n")
numeric_columns = df.select_dtypes(include=np.number).columns
print(numeric_columns)
# New line to count and print the total number of numeric columns
print("Number of numeric columns:", len(numeric_columns))

# for numeric data types that could be stored as string in the dataset
    # convert them into numeric type
for col in numeric_columns:    
    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# shows the number of countries in this dataset
print("unique country count :- ", df['country'].nunique())
# shows all the countries that are part of this dataset
print(df['country'].unique())

# counts frequency of each country - to check if any year could be missing
    # for a particular country
print(
    df['country']
    .value_counts()
)

# OUTLIER DETECTION FOR NUMERIC COLUMNS
print("---- outlier detection------")
outlier_flags = pd.DataFrame()

for col in numeric_columns:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_flags[col] = (
        (df[col] < lower) |
        (df[col] > upper)
    )

    
'''
The \(1.5 times IQR rule is a statistical convention introduced by 
mathematician John Tukey. It is used because it hits a "goldilocks" balance, 
closely matching the 3-sigma rule for normal distributions while remaining 
robust enough to handle skewed data without assuming a perfect bell curve
'''



outlier_counts = outlier_flags.sum()
print(outlier_counts)

# Creating a categorical column classifying countries as developing, developed and under-developed


# CREATE DEVELOPMENT STATUS CATEGORICAL COLUMN

# FIND GNI COLUMN AUTOMATICALLY
# =========================================================

print("\n----- Searching for GNI related columns -----\n")

gni_candidates = []

for col in df.columns:
    
    if 'gni' in col:
        
        gni_candidates.append(col)
        
        print(col)

# Use first matching GNI column
gni_column = gni_candidates[0]

print("\nSelected GNI Column :- ", gni_column)



# CONVERT GNI COLUMN TO NUMERIC

df[gni_column] = pd.to_numeric(
    df[gni_column],
    errors='coerce'
)

# CHECK MISSING VALUES IN GNI COLUMN
print("\nMissing values in GNI column :- ")
print(
    df[gni_column]
    .isnull()
    .sum()
)

# CREATE INCOME GROUP FUNCTION

# it is based on official guidelines given by World Bank

def income_group(gni):
    # Missing values
    if pd.isnull(gni):        
        return 'Unknown'
    # Low income countries
    elif gni < 1135:        
        return 'Low Income'
    # Lower middle income
    elif gni < 4466:        
        return 'Lower Middle Income'
    # Upper middle income
    elif gni < 13845:        
        return 'Upper Middle Income'
    # High income
    else:        
        return 'High Income'


# CREATE INCOME GROUP COLUMN

df['income_group'] = df[gni_column].apply(
    income_group
)


# CREATE DEVELOPMENT STATUS FUNCTION

def development_status(group):

    # Under-developed countries
    if group == 'Low Income':        
        return 'Under-developed'

    # Developing countries
    elif group in [
        'Lower Middle Income',
        'Upper Middle Income'
    ]:        
        return 'Developing'

    # Developed countries
    elif group == 'High Income':        
        return 'Developed'

    # Unknown values
    else:        
        return 'Unknown'



# CREATE FINAL CATEGORICAL COLUMN
df['development_status'] = df[
    'income_group'
].apply(development_status)



# VIEW RESULTS
print("\n----- Development Classification -----\n")
print(
    df[
        [
            'country',
            'year',
            gni_column,
            'income_group',
            'development_status'
        ]
    ].head(30)
)



# COUNT EACH CATEGORY
# KEEP ONLY MOST RECENT YEAR FOR EACH COUNTRY
# 1. First, get the latest year for every country (your existing logic)
latest_country_data = df.sort_values('year').drop_duplicates(
    subset='country',
    keep='last'
)

# 2. Then, filter that list to ONLY keep rows where the year is 2021
latest_country_data = latest_country_data[latest_country_data['year'] == 2021]

# INCOME GROUP COUNTS
print("\n----- Income Group Counts (Latest Year Only) -----\n")
print(
    latest_country_data['income_group']
    .value_counts()
)


# DEVELOPMENT STATUS COUNTS
print("\n----- Development Status Counts (Latest Year Only) -----\n")
print(
    latest_country_data['development_status']
    .value_counts()
)



# CORRECTED SAVE LINE:
# We add a filename in quotes as the first thing inside the parentheses
latest_country_data.to_csv("latest_year_country_data.csv", index=False)

print("\nSuccess! The file 'latest_year_global_econ_data.csv' has been saved to your folder.")

'''
1. UNDERDEVELOP, DEVELOPING, DEVELOPED COUNTRIES BASED ON RECENT YEAR DATA PUT IN
   OR LOAD TO ANOTHER CSV
2. START ANALYSING THEM AND GIVING GRAPH ON THEIR GROWTH ETC ALSO ANALYSE USING OTHER
    MACROECONOMIC PHENOMENON
3. THEN DO IT FOR WHOLE DATASET AS WELL AND GIVE COMPARISIONS AND DO EDA ANALYSIS ON THEM.
4. THEN CAN ALSO IF YOU WANT VISUALIZE THE SAME USING POWER BI AND COMPARE RESULTS
5. UPLOAD THIS ON GITHUB.

'''


