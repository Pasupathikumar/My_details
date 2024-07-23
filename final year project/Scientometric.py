# Import the necessary modules
import difflib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA

ca = CCA()

with open('2016 paper.txt') as file_1:
    file_1_text = file_1.readlines()
  
with open('2019 paper.txt') as file_2:
    file_2_text = file_2.readlines()
  

for line in difflib.unified_diff(
        file_1_text, file_2_text, fromfile='2016 paper.txt', 
        tofile='2019 paper.txt', lineterm=''):
    print(line)
#############################Machine Learning Algorithms ############################################# 
    
##############################1.Regression algorithms #################################################    

file_1_text = np.random.normal(170, 10, 250)
file_2_text = np.random.normal(170, 10, 250)
plt.hist(file_1_text)
plt.show() 
plt.hist(file_2_text)
plt.show() 
i = 2016
while i < 2017:
  print('Year==',i)
  if (i == 3):
    break
  i += 1
  
    
    
    
##############################################################################
#using csv datasets we are going to generate the graph for a given set of data froe about 10 years
data1 = pd.read_csv('accuracy dataset.csv')
  
df = pd.DataFrame(data1)
  
X = list(df.iloc[:, 0])
Y = list(df.iloc[:, 1])
  
plt.ylim(0,100)
plt.xlim(2010,2020)

# Plot the data using bar() method
plt.bar(X, Y, color='g')
plt.title("Accuracy over 10 years")
plt.xlabel("Years")
plt.ylabel("Accuracy")

max= df['Accuracy'].max()
print('accuracy:'+str(max))

# Show the plot
plt.show()

# analysing the dataset based on risk factors
data2 = pd.read_csv('risk factors dataset.csv')
  
df = pd.DataFrame(data2)
  
X = list(df.iloc[:, 0])
Y = list(df.iloc[:, 1])


plt.ylim(0,5)
plt.xlim(2010,2020)
  
# Plot the data using plot() method
plt.plot(X, Y, color='blue', linewidth = 3,
         marker='o', markerfacecolor='green', markersize=10)
plt.title("Risk factore over 10 years")
plt.xlabel("Years")
plt.ylabel("Risk Factors")
  
min= df['Risk factors'].min()
print('Risk factors:'+str(min))
# Show the plot
plt.show()
 


data3= pd.read_csv('Algorithms dataset.csv')
  
df = pd.DataFrame(data3)
  
X = list(df.iloc[:, 0])
Y = list(df.iloc[:, 1])

plt.xlim(2010,2020)
  
# Plot the data using bar() method
plt.plot(X, Y, color='cyan',linestyle='dotted', linewidth = 3,
         marker='o', markerfacecolor='black', markersize=10)
plt.title("No of algorithms used in projects over 10 years")
plt.xlabel("Years")
plt.ylabel(" Algorithms used")

# Show the plot
plt.show()


occur = df.groupby(['algorithms used']).size()
display(occur)

print('From the above we will be able to find which algorithm is most frequently used ')
print('Therfore the paper which has these characteristics can be used as reference by the authors who are going to work in future')

