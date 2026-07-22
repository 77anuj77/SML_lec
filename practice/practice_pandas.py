"""pandas is used for datamanipulation and data analysis.
dataframes are 2d labeled data structure like excels sheets
series are 1d labeled array capable of holding any data type
"""
import numpy as np
import pandas as pd

# creating df using csv ile
df1 = pd.read_csv(
    "/Users/ayushparoha/Documents/Data_Science/env/pandas/sales_data_sample.csv",
    encoding="latin1",
)
df1.head()

# conveting df to exel and csv file
df1 = {"name": ["ayush", "paroha"],
        "age": [22, 21],
        "city": ["indore", "ujjain"]
}
df1 = pd.DataFrame(df1)
df1
df1.to_csv("sample.csv")           
df1.to_excel("sample.xlsx", index=False)

# info method is used for summary of the dataframe
"""columns,rowa?
what type of ?
missing values?
memory used?"""
df1 = pd.read_csv("sample.csv")
df1.info()
df2 = pd.read_json("sample_Data.json")
df2.info()

# describe method used for geting statistical summary of numerical columns
"""mean std min max loe to up percentiles """
df1.describe()
df2.describe()

"""how big is your data 
what are the names of your columns"""
df1.shape
df1.columns

"""select specific columns
select specific rows
filter rows
combine multiplle conditions"""

print(df2.head())
df2["name"]
df2[["name", "price"]]
df2[df2["price"] > 500]  #SINGLE CONDITION
df2[(df2["price"] > 500) & (df2["category"] == "Electronics")]  #multiple confitions

data={
    "name":['Anuj','Shyam','Ghanshyam','Aditi','Ram','Govind','Radha','Ankush'],
    "age":[21,43,25,24,34,54,26,34],
    "salary":[50000,60000,45000,52000,49000,70000,48000,58000],
    "performance_score":[85,90,78,92,88,95,80,89]
}
df=pd.DataFrame(data)
print(df)
df1["bonus"]=df['salary']*0.1
df

#to get shape and columns name in a data
print(f"shape={df1.shape}, columns names:{df1.columns}")

#for selecting columns
df1["name"]
df1[["name", "age", "salary"]]
df1

#for inserting coulmns 
'''df.insert(loc, "column name", some_data)'''
df1.insert(0,"Employee_id",[10,23,45,2,12,4,5,89])
print(df1)
df1["allowance"]=df1["salary"]*0.2  #alternate method but not professional
print(df1)

#for updating specific value we use .l0c[]
df1.loc[1, "salary"]= 65000
print(df1)
list=["salary", "name"]
df1.loc[:,list].fillna()
#for updating full column we use same method as the creation of new column
df1["bonus"]=df1["salary"]*1.05
print(df1)

#removing elements fo rdeleting row and coulumns
df1.drop(columns=["allowance"], inplace=True)#it can be use for deleting multiple columns as well
print(df1)

'''handling of the missing data'''
data={
    "name":['Anuj',None,'Ghanshyam','Aditi','Ram','Govind','Radha','Ankush'],
    "age":[21,None,25,24,34,54,26,34],
    "salary":[50000,60000,45000,52000,49000,70000,48000,58000],
    "performance_score":[85,None,78,92,88,95,80,89]
}
df=pd.DataFrame(data)
print(df)
df.isnull()
df.isnull().sum()
print(df)

#for deleting rows and columns eith nan values we use df.dropna
df.dropna(axis= 0, inplace=True)
print(df)

#is data is important then we use dff.fillna
df.fillna(0,inplace=True)
print(df)

df["age"].fillna(df["age"].mean(), inplace=True)
df['salary'].fillna(df["salary"].mean(), inplace=True)
df['performance_score'].fillna(df["performance_score"].mean(), inplace=True)
print(df)

#intercolation techique for->> d.interpolate(method= __,axis=0, inplace=True)
'''preserves data intigrity
smooth values trends
no data loss'''

data={
    'time':[1,2,3,5,5],
    'value':[10,None,30,None,50]
}
df=pd.DataFrame(data)
print(df)
df['value']=df['value'].interpolate(method="linear")
print(df)

#sorting one and more columns
df.sort_values(by='performance_score', ascending=False,inplace=True)#one column
print(df)
df.sort_values(by=['age', 'performance_score'],ascending=True,inplace=True)
print(df)#also create a list for ascenting=[ true, False ] like that
df.sort_values(by=['age', 'performance_score'],ascending=[False,False],inplace=True)
print(df)
