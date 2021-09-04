# LendingClub-Loan-Status-Predictor
The model was built to determine whether a borrower will repay their loan. The model was built using historical data on loans given out with information whether or not the borrower defaulted. The model was built using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club.

# Imports and loading the Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../DATA/lending_club_loan_two.csv')
df.info()

![image](https://user-images.githubusercontent.com/89992872/132103153-6e4c3438-02ef-4b1a-8ec2-6865b6433e59.png)
