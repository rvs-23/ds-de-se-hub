# To run the code in your system, make the following changes
# Mandatory changes: hostname, username, password
# Optional changes: db_name, table_name

import pandas as pd
import mysql.connector as msc
from sqlalchemy import create_engine
import random

# The following are the MySQL connection settings
hostname='localhost'
username='root'
password='pass'
db_name = 'guessinggame'

def make_connection(hostname, username, password, db_name=None):
    '''
    Function to establish a database connection and create a database table.
    The database will be created only if it doesn't exist.
    Return -> connection variable
    '''
    # Making the connection with MySQL
    conn = msc.connect(host=hostname, user=username, password=password)
    
    # Making the connection with the database if df_name is provided
    if db_name is not None:
        query ='CREATE DATABASE IF NOT EXISTS '+str(db_name)
        cursor = conn.cursor()
        cursor.execute(query)
        conn = msc.connect(host=hostname, user=username, password=password, database=db_name)
        
    return conn

def make_table():
    '''
    Function to create a table in the format asked in the question.
    Return -> None
    '''
    conn_tab = make_connection(hostname, username, password, db_name)
    cursor = conn_tab.cursor()
    query_maketable='CREATE TABLE record(GameSessionNumber int,GamePlayNumber int, Guess int, SeekNumber int, GuessMatch int, PointsScored int);'
    cursor.execute(query_maketable)
    conn_tab.close()
    

def get_session():
    '''
    Function to fetch the maximum session number from the database of stored games.
    Return -> maximum session number (int)
    '''
    conn = make_connection(hostname,username,password,db_name)
    query = 'SELECT MAX(GameSessionNumber) FROM record;'
    cursor = conn.cursor()
    res = cursor.fetchall()[0][0]
    conn.close()
    return res

def guesssing_game_play():
    '''
    Function that contains the logic of the game.
    Return -> list of list where each nested list contains the record of each game play
    '''
    record = []
    game_play, guess, seek, match, points = 0, 0, 0, 0, 0    
    while True:
        guess = input("Choose a number between 1 and 20. Hit q or Q to quit. ")        
        if guess.lower()=='q':
            break
        else:
            seek = random.randint(1, 20)
            game_play += 1
            points = CalculatePoints(seek, int(guess))
            if points==5:
                match=1
            record.append([game_play, int(guess), seek, match, points])
    return record

def CalculatePoints(seek, guess):
    '''
    Function that calculates the scores for each game play.
    Return -> score (integer)
    '''
    # Stores the absolute difference between our guess and the computer's
    net = abs(seek-guess)
    # If the difference is 0, both values are same, return full score (5)
    if net==0:
        return 5
    # If the difference is 1 or 2 or 3, return a score of 3, 2 or 1
    elif net in (1, 2, 3):
        return 4-net
    # If the difference is greater than 4 and our guess is greater than computer, return -1
    elif seek<guess and net>=4:
        return -1
    else:
        return 0
    
if __name__ == '__main__':
    
    # flag variable to check if the connection is established for the first time
    flag = input('First time? Y/N ')
    # Make the table if the connection is first time, else no need
    if flag.lower() == 'y':
        make_table()
        
    # Play the game
    record = guesssing_game_play()
    # Create a DataFrame of the results
    df = pd.DataFrame(record, columns=['GamePlayNumber','Guess','SeekNumber','GuessMatch','PointsScored'])
    # Get the maximum session number from the table
    session=get_session()
    # For first session, make it 1 otherwise increment by 1
    if session is None:
        df['GameSessionNumber'] = 1
    else:
        df['GameSessionNumber'] = session+1
    
    # Create an engine and transfer the DataFrame to your server
    engine = create_engine("mysql+pymysql://" + username + ":" + password + "@" + hostname + "/" + db_name)
    df.to_sql('record',engine,if_exists='append',index=False)


##############################################################################


