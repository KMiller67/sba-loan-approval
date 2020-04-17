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

# Data cleaning, formatting and feature engineering
# Remove all records with a null MIS_Status (target field), 'Name', 'City', 'State', 'BankState', 'NewExist',
# 'RevLineCr', 'LowDoc' and 'DisbursementDate', since there isn't an easy value to fill them with and there are
# plenty of records left for analysis afterwards
SBA_data_copy.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist',
                             'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

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
        return x.replace('A', '')
    return x


SBA_data_copy['ApprovalFY'] = SBA_data_copy['ApprovalFY'].apply(clean_str).astype('int64')

# Change the type of NewExist to an integer, Zip and UrbanRural to str (categorical)
# and all currency-related fields to float values
SBA_data_copy = SBA_data_copy.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float',
                                      'BalanceGross': 'float', 'ChgOffPrinGr': 'float', 'GrAppv': 'float',
                                      'SBA_Appv': 'float'})

# Create a new column with the industry the NAICS code represents
SBA_data_copy['Industry'] = SBA_data_copy['NAICS'].astype('str').apply(lambda x: x[:2])
SBA_data_copy['Industry'] = SBA_data_copy['Industry'].map({
                            '11': 'Ag/For/Fish/Hunt',
                            '21': 'Min/Quar/Oil_Gas_ext',
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

# Remove records where Industry is NaN (NAICS code was a 0)
SBA_data_copy.dropna(subset=['Industry'], inplace=True)

# Create flag column IsFranchise based on FranchiseCode column
SBA_data_copy.loc[(SBA_data_copy['FranchiseCode'] <= 1), 'IsFranchise'] = 0
SBA_data_copy.loc[(SBA_data_copy['FranchiseCode'] > 1), 'IsFranchise'] = 1

# Adjust current flag identifiers for NewExist, RevLineCr, LowDoc, and MIS_Status
# NewExist
# Make sure NewExist has only 1s and 2s; Remove records where NewExist isn't 1 or 2
SBA_data_copy['NewExist'].unique()
SBA_data_copy = SBA_data_copy[(SBA_data_copy['NewExist'] == 1) | (SBA_data_copy['NewExist'] == 2)]

# NewExist - 0 = Existing business, 1 = New business; Renamed to NewBusiness
SBA_data_copy.loc[(SBA_data_copy['NewExist'] == 1), 'NewBusiness'] = 0
SBA_data_copy.loc[(SBA_data_copy['NewExist'] == 2), 'NewBusiness'] = 1

# RevLineCr and LowDoc
# Double check RevLineCr and LowDoc unique values
SBA_data_copy['RevLineCr'].unique()
SBA_data_copy['LowDoc'].unique()

# Remove records where RevLineCr != 'Y' or 'N' and LowDoc != 'Y' or 'N'
SBA_data_copy = SBA_data_copy[(SBA_data_copy['RevLineCr'] == 'Y') | (SBA_data_copy['RevLineCr'] == 'N')]
SBA_data_copy = SBA_data_copy[(SBA_data_copy['LowDoc'] == 'Y') | (SBA_data_copy['LowDoc'] == 'N')]

# RevLineCr and LowDoc - 0 = No, 1 = Yes
SBA_data_copy['RevLineCr'] = np.where(SBA_data_copy['RevLineCr'] == 'N', 0, 1)
SBA_data_copy['LowDoc'] = np.where(SBA_data_copy['LowDoc'] == 'N', 0, 1)

# Convert ApprovalDate and DisbursementDate columns to datetime values
# ChgOffDate not changed to datetime since it is not of value and will be removed later
SBA_data_copy[['ApprovalDate', 'DisbursementDate']] = SBA_data_copy[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)

# Create DaysToDisbursement column which calculates the number of days passed between DisbursementDate and ApprovalDate
# Some businesses may be in more urgent need of funds and the length of time it take to receive them could affect their
# ability to repay
SBA_data_copy['DaysToDisbursement'] = SBA_data_copy['DisbursementDate'] - SBA_data_copy['ApprovalDate']

# Change DaysToDisbursement from a timedelta64 dtype to an int64 dtype
# Converts series to str, removes all characters after the space before 'd' in days for each record, then changes
# the dtype to int
SBA_data_copy['DaysToDisbursement'] = SBA_data_copy['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d')-1]).astype('int64')

# Create StateSame flag field which identifies where the business State is the same as the BankState
SBA_data_copy['StateSame'] = np.where(SBA_data_copy['State'] == SBA_data_copy['BankState'], 1, 0)

# Create SBA_AppvPct field since the guaranteed amount is based on a percentage of the gross loan amount
# rather than dollar amount in most situations
SBA_data_copy['SBA_AppvPct'] = SBA_data_copy['SBA_Appv']/SBA_data_copy['GrAppv']

# Create AppvDisbursed flag field signifying if the loan amount disbursed was equal to the full amount approved
SBA_data_copy['AppvDisbursed'] = np.where(SBA_data_copy['DisbursementGross'] == SBA_data_copy['GrAppv'], 1, 0)

# Change Target field (MIS_Status) so P I F = 1 and CHGOFF = 0 so we can see what contributes to a successful loan
SBA_data_copy['MIS_Status'] = np.where(SBA_data_copy['MIS_Status'] == 'P I F', 1, 0)

# Format dtypes where necessary after feature engineering
SBA_data_copy = SBA_data_copy.astype({'RevLineCr': 'int64', 'LowDoc': 'int64', 'MIS_Status': 'int64',
                                      'IsFranchise': 'int64', 'NewBusiness': 'int64', 'StateSame': 'int64'})

SBA_data_copy['EarlyDisbursement'] = np.where(SBA_data_copy['DisbursementDate'] < SBA_data_copy['ApprovalDate'], 1, 0)

# Remove unnecessary columns
# LoanNr_ChkDgt and Name provide no value to the analysis
# ChgOffDate only applies when a loan is charged off and isn't relevant to the analysis either
# NAICS replaced by Industry; NewExist replaced by NewBusiness flag; FranchiseCode replaced by IsFranchise flag
# ApprovalDate and DisbursementDate dropped - hypothesis that DaysToDisbursement will be more valuable
# SBA_Appv since guaranteed amount is based on a percentage of gross loan amount, not dollar amount
SBA_data_copy.drop(columns=['LoanNr_ChkDgt', 'Name', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
                            'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv'], inplace=True)

# Verify all null values are removed from data
SBA_data_copy.isnull().sum()

# Only look at records with an ApprovalFY of at least 2010 for a more relevant analysis
SBA_data_copy = SBA_data_copy[SBA_data_copy['ApprovalFY'] >= 2010]

# Check how many records and fields are remaining
SBA_data_copy.shape

# Get some information about the data
SBA_data_copy.describe()

# NOTES:
# A vast majority of the records selected were approved during or before 2011
# 75% of the businesses applying for loans in this sample had <= 10 employees, created <= 2 jobs and retained <= 9 jobs
# The average loan term for the sample was about 7.5 years (90 months), with 75% of loans being 87 months or less
# None of the loans in the sample were a part of the LowDoc loan program, so this field can be removed
# Mean DisbursementGross is about $275,000 with std of $510,000 and 75% of loans being <= about $287,000
# Mean GrAppv is about $250,000 with std of $482,000 and 75% of loans being <= $250,000
# Mean MIS_Status is about 0.91, suggesting about 91% of the records were paid in full; will need to be accounted for
# Vast majority of sampled loans are for small businesses that are not franchises
# About 29% of sampled loans are new businesses
# DaysToDisbursement was about 44 days on average, however the mean was -16 days and max was 1644 days so we'll need to
# check for outliers
# About 54% of loans were from a bank that was in the same state as the applying business
# The average percentage of guaranteed loan amount was about 66%
# About 68% of loans disbursed the full amount of the loan that was originally approved

# Things to explore:
# Average loan amount by industry
#

# Remove LowDoc field
SBA_data_copy.drop(columns='LowDoc', inplace=True)

# Create flag to signify if a larger amount was disbursed than what the Bank had approved
# Likely RevLineCr?
SBA_data_copy['DisburseGreaterAppv'] = np.where(SBA_data_copy['DisbursementGross'] > SBA_data_copy['GrAppv'], 1, 0)

# Exploring with Visualizations
# Create a groupby object on Industry for use in visualization
industry_group = SBA_data_copy.groupby(['Industry'])

# Data frames based on groupby on Industry looking at aggregate and average values
df_industrySum = industry_group.sum().sort_values('DisbursementGross', ascending=False)
df_industryAve = industry_group.mean().sort_values('DisbursementGross', ascending=False)

# Add subplots to figure to build 1x2 grid and specify position of each subplot
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)

fig, ax = plt.subplots()
ax.bar(df_industrySum.index, df_industrySum['DisbursementGross']/1000000000)
ax.set_xticklabels(df_industrySum.index, rotation=60, horizontalalignment='right', fontsize=6)

ax.set_title('Gross SBA Loan Disbursement by Industry from 2010-2014', fontsize=15)
ax.set_xlabel('Industry')
ax.set_ylabel('Gross Loan Disbursement (Billions)')

plt.show()
