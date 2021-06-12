import numpy as np
import pandas as pd

dataset = pd.read_csv('SentimentDataset.csv') # Get dataset with features
newtwt = pd.read_csv('New_Tweet.csv') # Get a new tweet as a file that has features

custom_dataset = dataset.iloc[:, 3:] # Get after 3th columns which are only feature
custom_newtwt = newtwt.iloc[:, 3:] 
result = pd.DataFrame(columns=['Sentimental', 'cos_sim']) # Create a table to only store values of sentimental and similarity


for index, row in custom_dataset.iterrows(): # Calculate the similarity of the new tweet with each row of dataset
    dot = np.dot(row.to_numpy(), custom_newtwt.values[0]) # calculate the numerator
    norm_current_row = np.linalg.norm(row.to_numpy()) # calculate the denominator using linear algebra function
    norm_newtwt = np.linalg.norm(custom_newtwt.values[0])
    if (norm_current_row == 0 or norm_newtwt == 0): 
        cos = 0 
    else:
        cos = dot / (norm_current_row * norm_newtwt)  
    result = result.append({'Sentimental': dataset.at[index,"Sentimental"], 'cos_sim': cos},ignore_index=True) # fill the table via similarities


# colors for beautiful prints ;-)))
red = "\033[1;31m" 
green = "\033[032m"
bold = "\033[;1m"
reset = "\033[0;0m"
print(bold, red, "Original Sentiment: " , newtwt.iloc[0]['Sentimental']) 


Pos = 0
Neg = 0
sort_result = result.sort_values(by='cos_sim',ascending=False) # Sort by more similarity 
temp = sort_result.head(50) # consider K = 50
print(reset, 'K=50 nearest neighbors')
for i, data in temp.iterrows(): # count positives and negatives
        if data[0]=="Positive":
            Pos = Pos + 1
        else:
            Neg = Neg + 1 
print(reset, 'Positive Side: ', Pos)
print(reset, 'Negative Side: ', Neg)


if Pos > Neg: # check for positive and negative
    print(bold, green,'Answer is: Positive')
elif Pos < Neg:
    print(bold, green,'Answer is: Negative')
else:
    print('can not predict due to equality')
    
print(reset)