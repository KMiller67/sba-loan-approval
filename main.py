from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # Load data using Path, making it easy to use on different operating systems
    SBA_data = pd.read_csv(Path().joinpath('should-this-loan-be-approved-or-denied', 'SBAnational.csv'))

    # Copies data frame to use for exploration purposes
    SBA_data_copy = SBA_data.copy()

    # Data cleaning, formatting and feature engineering
    # Remove all records with a null value
    SBA_data_copy.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist',
                                 'RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)

    # Remove '$', commas, and extra spaces from records in columns with dollar values that should be floats
    SBA_data_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = \
        SBA_data_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(
            lambda x: x.strip().replace('$', '').replace(',', ''))

    # A few records in ApprovalFY are listed as 1976A; Remove the 'A' from these records for ease of formatting
    # ApprovalFY has multiple data types
    SBA_data_copy['ApprovalFY'].apply(type).value_counts()
    SBA_data_copy['ApprovalFY'] = SBA_data_copy['ApprovalFY'].apply(clean_str).astype('int64')

    # Change the type of NewExist to an integer, Zip and UrbanRural to str (categorical)
    # and all currency-related fields to float values
    SBA_data_copy = SBA_data_copy.astype(
        {'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float',
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

    # Make Default target field based on MIS_Status so P I F = 1 and CHGOFF = 0
    # Allows us to see what features are prevalent in a defaulted loan
    SBA_data_copy['Default'] = np.where(SBA_data_copy['MIS_Status'] == 'P I F', 0, 1)

    # Convert ApprovalDate and DisbursementDate columns to datetime values
    # ChgOffDate not changed to datetime since it is not of value and will be removed later
    SBA_data_copy[['ApprovalDate', 'DisbursementDate']] = SBA_data_copy[['ApprovalDate', 'DisbursementDate']].apply(
        pd.to_datetime)

    # Create DisbursementFY field for time selection criteria later
    SBA_data_copy['DisbursementFY'] = SBA_data_copy['DisbursementDate'].map(lambda x: x.year)

    # Create DaysToDisbursement column which calculates the number of days passed between DisbursementDate and
    # ApprovalDate
    SBA_data_copy['DaysToDisbursement'] = SBA_data_copy['DisbursementDate'] - SBA_data_copy['ApprovalDate']

    # Change DaysToDisbursement from a timedelta64 dtype to an int64 dtype
    # Converts series to str, removes all characters after the space before 'd' in days for each record, then changes
    # the dtype to int
    SBA_data_copy['DaysToDisbursement'] = SBA_data_copy['DaysToDisbursement'].astype('str').apply(
        lambda x: x[:x.index('d') - 1]).astype('int64')

    # Create StateSame flag field which identifies where the business State is the same as the BankState
    SBA_data_copy['StateSame'] = np.where(SBA_data_copy['State'] == SBA_data_copy['BankState'], 1, 0)

    # Create SBA_AppvPct field since the guaranteed amount is based on a percentage of the gross loan amount
    # rather than dollar amount in most situations
    SBA_data_copy['SBA_AppvPct'] = SBA_data_copy['SBA_Appv'] / SBA_data_copy['GrAppv']

    # Create AppvDisbursed flag field signifying if the loan amount disbursed was equal to the full amount approved
    SBA_data_copy['AppvDisbursed'] = np.where(SBA_data_copy['DisbursementGross'] == SBA_data_copy['GrAppv'], 1, 0)

    # Format dtypes where necessary after feature engineering
    SBA_data_copy = SBA_data_copy.astype({'RevLineCr': 'int64', 'LowDoc': 'int64', 'Default': 'int64',
                                          'IsFranchise': 'int64', 'NewBusiness': 'int64', 'StateSame': 'int64',
                                          'AppvDisbursed': 'int64'})

    # Field for loans backed by Real Estate (loans with a term of at least 20 years)
    SBA_data_copy['RealEstate'] = np.where(SBA_data_copy['Term'] >= 240, 1, 0)

    # Field for loans active during the Great Recession (2007-2009)
    SBA_data_copy['GreatRecession'] = np.where(((2007 <= SBA_data_copy['DisbursementFY']) &
                                                (SBA_data_copy['DisbursementFY'] <= 2009)) |
                                               ((SBA_data_copy['DisbursementFY'] < 2007) &
                                                (SBA_data_copy['DisbursementFY'] +
                                                 (SBA_data_copy['Term']/12) >= 2007)), 1, 0)

    # Remove unnecessary columns
    SBA_data_copy.drop(columns=['LoanNr_ChkDgt', 'Name', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
                                'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv'],
                       inplace=True)

    # Only look at records with a DisbursementFY through 2010
    SBA_data_copy = SBA_data_copy[SBA_data_copy['DisbursementFY'] <= 2010]

    # Edit to keep LowDoc field since new time frame has more LowDoc records
    # Remove LowDoc field
    SBA_data_copy.drop(columns='LowDoc', inplace=True)

    # Create flag to signify if a larger amount was disbursed than what the Bank had approved
    SBA_data_copy['DisbursedGreaterAppv'] = np.where(SBA_data_copy['DisbursementGross'] > SBA_data_copy['GrAppv'], 1, 0)

    # Remove records with loans disbursed prior to being approved
    SBA_data_copy = SBA_data_copy[SBA_data_copy['DaysToDisbursement'] >= 0]

    # Correlation Matrix
    corr_matrix(df=SBA_data_copy)

    # Exploring with Visualizations
    # Total/Average disbursed loan amount by industry
    # Create a groupby object on Industry for use in visualization
    industry_group = SBA_data_copy.groupby(['Industry'])

    # Data frames based on groupby Industry looking at aggregate and average values
    df_industrySum = industry_group.sum().sort_values('DisbursementGross', ascending=False)
    df_industryAve = industry_group.mean().sort_values('DisbursementGross', ascending=False)

    # Establish figure for placing bar charts side-by-side
    fig = plt.figure(figsize=(20, 5))

    # Add subplots to figure to build 1x2 grid and specify position of each subplot
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Bar chart 1 = Gross SBA Loan Disbursement by Industry
    ax1.bar(df_industrySum.index, df_industrySum['DisbursementGross'] / 1000000000)
    ax1.set_xticklabels(df_industrySum.index, rotation=30, horizontalalignment='right', fontsize=6)

    ax1.set_title('Gross SBA Loan Disbursement by Industry from 2010-2014', fontsize=15)
    ax1.set_xlabel('Industry')
    ax1.set_ylabel('Gross Loan Disbursement (Billions)')

    # Bar chart 2 = Average SBA Loan Disbursement by Industry
    ax2.bar(df_industryAve.index, df_industryAve['DisbursementGross'])
    ax2.set_xticklabels(df_industryAve.index, rotation=30, horizontalalignment='right', fontsize=6)

    ax2.set_title('Average SBA Loan Disbursement by Industry through 2010', fontsize=15)
    ax2.set_xlabel('Industry')
    ax2.set_ylabel('Average Loan Disbursement')

    plt.show()

    # Average days to disbursement by industry
    fig2, ax = plt.subplots()

    ax.bar(df_industryAve.index, df_industryAve['DaysToDisbursement'].sort_values(ascending=False))
    ax.set_xticklabels(df_industryAve['DaysToDisbursement'].sort_values(ascending=False).index, rotation=35,
                       horizontalalignment='right', fontsize=6)

    ax.set_title('Average Days to SBA Loan Disbursement by Industry through 2010', fontsize=15)
    ax.set_xlabel('Industry')
    ax.set_ylabel('Average Days to Disbursement')

    plt.show()

    # Paid in full and defaulted loans
    fig3 = plt.figure(figsize=(15, 10))

    ax1a = plt.subplot(2, 1, 1)
    ax2a = plt.subplot(2, 1, 2)

    # Number of PIF and Defaulted loans by industry
    stacked_setup(df=SBA_data_copy, col='Industry', axes=ax1a)
    ax1a.set_xticklabels(SBA_data_copy.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                         rotation=35, horizontalalignment='right', fontsize=6)

    ax1a.set_title('Number of PIF/Defaulted Loans by Industry')
    ax1a.set_xlabel('Industry')
    ax1a.set_ylabel('Number of PIF/Defaulted Loans')
    ax1a.legend()

    # Number of PIF and Defaulted loans by State
    stacked_setup(df=SBA_data_copy, col='State', axes=ax2a)

    ax2a.set_title('Number of PIF/Defaulted Loans by State')
    ax2a.set_xlabel('State')
    ax2a.set_ylabel('Number of PIF/Defaulted Loans')
    ax2a.legend()

    plt.tight_layout()
    plt.show()

    # Paid in full and Defaulted loans by DisbursementFY
    fig4, ax4 = plt.subplots(figsize=(15, 5))

    stack_data = SBA_data_copy.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')
    x = stack_data.index
    y = [stack_data[1], stack_data[0]]

    ax4.stackplot(x, y, labels=['Default', 'Paid in full'])
    ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
    ax4.set_xlabel('Disbursement Year')
    ax4.set_ylabel('Number of PIF/Defaulted Loans')
    ax4.legend(loc='upper left')

    plt.show()


def clean_str(rec) -> str:
    """
    Applies formatting to records of str type only
    :param rec: Record in specified pd.Series
    :return: str
    """
    if isinstance(rec, str):
        return rec.replace('A', '')
    return rec


def corr_matrix(df: pd.DataFrame):
    """
    Creates a correlation matrix based on supplied data frame.
    :param df: Data frame correlation matrix is based on.
    :return: Correlation matrix of provided data.
    """
    matrix = df.corr()
    sns.heatmap(matrix, annot=True)
    plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()


def stacked_setup(df: pd.DataFrame, col: pd.Series, axes, stack_col: str = 'Default'):
    """
    Creates stacked bar charts comparing loans in Default to loans Paid in Full, grouped by desired column.
    :param df: Original data frame
    :param col: X-axis grouping
    :param axes: axis of plot figure to apply changes to
    :param stack_col: Column used for 'stacking' data; defaults to 'Default' column
    :return: Stacked bar chart
    """
    data = df.groupby([col, stack_col])[col].count().unstack(stack_col)

    axes.bar(data.index, data[1], label='Default')
    axes.bar(data.index, data[0], bottom=data[1], label='Paid in full')


if __name__ == '__main__':
    main()
