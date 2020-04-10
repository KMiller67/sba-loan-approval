from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data using Path, making it easy to use on different operating systems
SBA_data = pd.read_csv(Path().joinpath('should-this-loan-be-approved-or-denied', 'SBAnational.csv'))

# Copies data frame to use for exploration purposes
SBA_data_copy = SBA_data.copy()

# Check for null values
SBA_data_copy.isnull().sum()

# Remove all records with a null MIS_Status (target field), 'Name', 'City', 'State', 'BankState', 'NewExist',
# 'RevLineCr', 'LowDoc' and 'DisbursementDate', since there isn't an easy value to fill them with and there are plenty
# of records left for analysis afterwards
SBA_data_copy.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist',
                             'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace= True)

# Check the types of the number of null values remaining in other columns
SBA_data_copy.isnull().sum()
SBA_data_copy.dtypes

# Remove '$', commas, and extra spaces from records in columns with dollar values that should be floats
SBA_data_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = \
    SBA_data_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']]\
        .applymap(lambda x: x.strip().replace('$', '').replace(',', ''))

# A few records in ApprovalFY are listed as 1976A; Remove the 'A' from these records for ease of formatting
# ApprovalFY has multiple data types
SBA_data_copy['ApprovalFY'].apply(type).value_counts()

# Create a function to apply formatting to the records of str type only
def clean_str(x):
    if isinstance(x, str):
        return(x.replace('A', ''))
    return(x)

SBA_data_copy['ApprovalFY'] = SBA_data_copy['ApprovalFY'].apply(clean_str).astype('int64')

# Change the type of NewExist to an integer and all currency-related fields to float values
SBA_data_copy = SBA_data_copy.astype({'NewExist':'int64', 'DisbursementGross':'float64', 'BalanceGross': 'float64',
                      'ChgOffPrinGr':'float64', 'GrAppv': 'float64', 'SBA_Appv':'float64'})

# Create a new column with the industry the NAICS code represents
SBA_data_copy['Industry'] = SBA_data_copy['NAICS'].astype('str').apply(lambda x: x[:2])
SBA_data_copy['Industry'] = SBA_data_copy['Industry'].map({
                            '11':'Ag/For/Fish/Hunt',
                            '21':'Min/Quar/Oil_Gas_ext',
                            '22': 'Utilities',
                            '23': 'Construction',
                            '31': 'Manufacturing',
                            '32': 'Manufacturing',
                            '33': 'Manufacturing',
                            '42': 'Wholesale_trade',
                            '44': 'Retail_trade',
                            '45': 'Retail_trade',
                            '48': 'Trans/Ware',
                            '49': 'Trans/Ware',
                            '51': 'Information',
                            '52': 'Finance/Insurance',
                            '53': 'RE/Rental/Lease',
                            '54': 'Prof/Science/Tech',
                            '55': 'Mgmt_comp',
                            '56': 'Admin_sup/Waste_Mgmt_Rem',
                            '61': 'Educational',
                            '62': 'Healthcare/Social_assist',
                            '71': 'Arts/Entertain/Rec',
                            '72': 'Accom/Food_serv',
                            '81': 'Other_no_pub',
                            '92': 'Public_Admin'
                            })

# Create flag column IsFranchise based on FranchiseCode column
SBA_data_copy.loc[(SBA_data_copy['FranchiseCode'] <= 1), 'IsFranchise'] = '0'
SBA_data_copy.loc[(SBA_data_copy['FranchiseCode'] > 1), 'IsFranchise'] = '1'

# Remove unnecessary columns
# LoanNr_ChkDgt provides no value to the analysis
# ChgOffDate only applies when a loan is charged off and isn't relevant to the analysis either
# NAICS is no longer needed as we have created Industry to replace it
SBA_data_copy.drop(columns=['LoanNr_ChkDgt', 'NAICS', 'FranchiseCode', 'ChgOffDate'], inplace=True)

# Only look at records with an ApprovalFY of at least 2010 for a more relevant analysis
# A few records had an NAICS of 0 and were mapped as NaN, so those records are removed as well
SBA_data_copy = SBA_data_copy[(SBA_data_copy['ApprovalFY'] >= 2010) & (SBA_data_copy['Industry'].notnull())]

# Check how many records are remaining
SBA_data_copy.shape
