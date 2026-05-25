# Update the hostname, username, password and df_name as per
# your MySQl settings

import pandas as pd
import mysql.connector as msc
from sqlalchemy import create_engine

print("Reading the csv ...")
df = pd.read_csv('merged_crop_data.csv')
df = df.round(4)

hostname='localhost'
username='root'
password='pass'
db_name = 'crop'

print("Creating functions to create the connection and the database...")
def make_connection(hostname,username,password):
    conn = msc.connect(host=hostname, user=username,password=password)
    return conn

def make_db(hostname, username, password, conn, db_name):
    query='CREATE DATABASE IF NOT EXISTS '+str(db_name)
    cursor = conn.cursor()
    cursor.execute(query)
    conn = msc.connect(host=hostname, user=username, password=password, database=db_name)
    return conn

print("Making the connection...")
conn = make_connection(hostname, username, password)
print("Connection established...")
print(f"Creating databse {db_name}")
conn = make_db(hostname, username, password, conn, db_name)
print("Database created...")

cursor = conn.cursor()
print("Creating table to store the dataframe...")
query_maketable = 'CREATE TABLE cerealcropyielddata(CountryCode nvarchar(3),CountryDescription nvarchar(50), Decade smallint, IncompleteWarning char(1), BarleyActualYield decimal(8,6),RyeActualYield decimal(8,6),WheatActualYield decimal(8,6),FertilizerKgHa decimal(8,6),PesticideKgHa decimal(8,6), InsertedBy nvarchar(50), PRIMARY KEY(CountryCode,Decade))'
cursor.execute(query_maketable)

print("Sending the data to the SQL server...")
engine = create_engine("mysql+pymysql://" + username + ":" + password + "@" + hostname + "/" + db_name)
df.to_sql('cerealcropyielddata', engine, if_exists='replace', index=False)
print("Done...")

###############################################################################

