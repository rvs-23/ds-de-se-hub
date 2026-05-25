import pandas as pd
import os
import numpy as np

print("Specifying the base path and the location of the 3 filenames relative to the base path...")
base_path = os.path.dirname(os.path.realpath('__file__'))
in_filename_crop = '..\\cropyield.csv'
in_filename_fert = '..\\fertilizers.csv'
in_filename_pest = '..\\pesticide-use-per-hectare-of-cropland.csv'

print("Reading the crop, fertilizer and pesticide csv's... ")
df_crop = pd.read_csv(os.path.join(base_path, in_filename_crop))
df_fert = pd.read_csv(os.path.join(base_path, in_filename_fert))
df_pest = pd.read_csv(os.path.join(base_path, in_filename_pest))

print("Merging the fertilizer and pesticide csv on Entity and Year... ")
df_fert_pest = pd.merge(df_fert, df_pest, on=['Entity', 'Year'], suffixes=['', '_pest'])
# Dropping the extra Code column
df_fert_pest.drop(columns=['Code_pest'], inplace=True)
print("Merging the crop csv with the fertilizer pesticide merged csv... ")
df_fert_pest_crop = pd.merge(df_fert_pest, df_crop, on=['Entity', 'Year'])

print("Calculating the actual yields of barley, rice and wheat...")
df_fert_pest_crop['barley_actual_yield'] = df_fert_pest_crop['barley_attainable'] - df_fert_pest_crop['barley_yield_gap']
df_fert_pest_crop['rye_actual_yield'] = df_fert_pest_crop['rye_attainable'] - df_fert_pest_crop['rye_yield_gap']
df_fert_pest_crop['wheat_actual_yield'] = df_fert_pest_crop['wheat_attainable'] - df_fert_pest_crop['wheat_yield_gap']
# Changing the Year column to appropriate datatype
df_fert_pest_crop['Year'] = pd.to_datetime(df_fert_pest_crop['Year'], format='%Y')

# Creating a list of required columns
columns_req = ['Entity', 'Code',
               'Year', 'Fertilizer consumption (kilograms per hectare of arable land)',
               'Pesticides indicators - Pesticides (total) - 1357 - Use per area of cropland - 5159 - kg/ha',
               'rye_actual_yield', 'barley_actual_yield',
               'wheat_actual_yield']

print("Filtering the csv by selecting only the required columns...")
df_final = df_fert_pest_crop[columns_req].copy()

# Defining aggregation technique for each column for grouping
agg = {columns_req[0]: 'first', columns_req[1]: 'first',
       columns_req[2]: np.mean, columns_req[3]: np.mean,
       columns_req[4]: np.mean, columns_req[4]: np.mean,
       columns_req[5]: np.mean, columns_req[6]: np.mean,
       columns_req[7]: np.mean
       }

# Creating an InsertedBy variable to specify the person inserting (Database requirement)
InsertedBy = "xyz"

print("Creating an empty DataFrame to store the final result...")
df_result = pd.DataFrame()

print("Grouping begins...")
# Iterating over each country and grouping the result by decade
for code in df_final['Code'].unique():
    temp = df_final[df_final['Code']==code]
    temp = temp.groupby([pd.Grouper(key='Year', freq='10Y')]).aggregate(agg)
    df_result = pd.concat([df_result, temp], axis=0)
print("Groupig done...")

# Dropping the extra column
df_result.drop(columns=['Year'], inplace=True)
df_result.reset_index(inplace=True)    

# Extracting the Year from the timestamp
df_result['Year'] = df_result['Year'].dt.year
# Converting the Year into decade as per requirement and choosing the last two digits
df_result['Year'] = df_result.Year.apply(lambda x: x-10)
df_result['Year'] = df_result.Year.apply(lambda x: str(x)[2:])

# Updating the column names as per the database requirement
df_result.columns = ['Decade', 'CountryDescription', 'CountryCode',
                     'FertilizerKgHa', 'PesticideKgHa', 'RyeActualYield',
                     'BarleyActualYield', 'WheatActualYield',
                     ]

print("Creating Incomplete Waring for missing data...")
df_result['IncompleteWarning'] = df_result.isnull().any(axis=1)
df_result['IncompleteWarning'] = np.where(df_result['IncompleteWarning']==True, 'Y', 'N')
df_result['InsertedBy'] = InsertedBy

print("Completed. Saving the final csv ...")
df_result.to_csv('merged_crop_data.csv', index=False, encoding='utf-8')
print("Saving done...")

###############################################################################

# Output:

# Specifying the base path and the location of the 3 filenames relative to the base path...
# Reading the crop, fertilizer and pesticide csv's... 
# Merging the fertilizer and pesticide csv on Entity and Year... 
# Merging the crop csv with the fertilizer pesticide merged csv... 
# Calculating the actual yields of barley, rice and wheat...
# Filtering the csv by selecting only the required columns...
# Creating an empty DataFrame to store the final result...
# Grouping begins...
# Groupig done...
# Creating Incomplete Waring for missing data...
# Completed. Saving the final csv ...
# Saving done...