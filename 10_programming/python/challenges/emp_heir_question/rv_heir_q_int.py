"""
The code below solves the heirarchy question without using self joins. The output
obtained with this code is attached at the end.

"""

import os
import pandas as pd

base_path = os.path.dirname(os.path.realpath(__file__))

df_cemp = pd.read_excel(os.path.join(base_path, '../Downloads/client_emp.xlsx'))
df_emng = pd.read_excel(os.path.join(base_path, '../Downloads/emp_mng.xlsx'))

# Making employee code the index for easy lookup using .loc
df_emng.set_index('EMP', inplace=True)

# printing both the tables used
print(df_cemp, "\n")
print(df_emng, "\n")

a = []
for idx, rows in df_cemp.iterrows():
    
    # emp -> contains the current employee
    # heir -> stores the heirarchy that exists for a particular client
    emp = rows['EMP']
    heir, mng = [], ''
    
    # Keep looking up managers until they cannot be found anymore
    # Works for endless heirarchy
    while mng is not None:
        try:
            mng = df_emng.loc[emp]['manager']
            print(f"For emp {emp} manager {mng}")
            
            # The manager now becomes the employee for the next iteration
            emp = mng
            heir.append(mng)
        except:
            mng = None
            
    a.append(heir)

print()
df_cemp['Heirarchy'] = a
df_cemp['Heirarchy'] = df_cemp['Heirarchy'].apply(lambda x: ', '.join(x))
print(df_cemp)
    

# -----------------------------------------------------------------------------

#       CLIENT EMP
# 0    CLIENT1  X1
# 1    CLIENT2  Y1
# 2    CLIENT3  Z1
# 3    CLIENT4  X8
# 4    CLIENT5  X7
# 5    CLIENT6  X6
# 6    CLIENT7  X5
# 7    CLIENT8  X4
# 8    CLIENT9  Z1
# 9   CLIENT10  X8
# 10  CLIENT11  X7 

#     manager
# EMP        
# X1       X2
# Y1       X2
# Z1       X2
# X8       T1
# X7       T1
# X6       T1
# X5       T2
# X4       T2
# T2       X3
# T1       X3
# X2       X3 

# For emp X1 manager X2
# For emp X2 manager X3
# For emp Y1 manager X2
# For emp X2 manager X3
# For emp Z1 manager X2
# For emp X2 manager X3
# For emp X8 manager T1
# For emp T1 manager X3
# For emp X7 manager T1
# For emp T1 manager X3
# For emp X6 manager T1
# For emp T1 manager X3
# For emp X5 manager T2
# For emp T2 manager X3
# For emp X4 manager T2
# For emp T2 manager X3
# For emp Z1 manager X2
# For emp X2 manager X3
# For emp X8 manager T1
# For emp T1 manager X3
# For emp X7 manager T1
# For emp T1 manager X3

#       CLIENT EMP Heirarchy
# 0    CLIENT1  X1    X2, X3
# 1    CLIENT2  Y1    X2, X3
# 2    CLIENT3  Z1    X2, X3
# 3    CLIENT4  X8    T1, X3
# 4    CLIENT5  X7    T1, X3
# 5    CLIENT6  X6    T1, X3
# 6    CLIENT7  X5    T2, X3
# 7    CLIENT8  X4    T2, X3
# 8    CLIENT9  Z1    X2, X3
# 9   CLIENT10  X8    T1, X3
# 10  CLIENT11  X7    T1, X3
