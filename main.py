from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def main():

    # Load data using Path, making it easy to use on different operating systems
    df = pd.read_csv(Path().joinpath('should-this-loan-be-approved-or-denied', 'SBAnational.csv'))

    # Copies data frame to use for exploration purposes
    df_copy = df.copy()

    # Data cleaning, formatting and feature engineering
    # Remove all records with a null value
    df_copy.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate',
                           'MIS_Status'], inplace=True)

    # Remove '$', commas, and extra spaces from records in columns with dollar values that should be floats
    df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = \
        df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(
        lambda record: record.strip().replace('$', '').replace(',', ''))

    # A few records in ApprovalFY are listed as 1976A; Remove the 'A' from these records for ease of formatting
    # ApprovalFY has multiple data types
    df_copy['ApprovalFY'] = df_copy['ApprovalFY'].apply(clean_str).astype('int64')

    # Change the type of NewExist to an integer, Zip and UrbanRural to str (categorical)
    # and all currency-related fields to float values
    df_copy = df_copy.astype(
        {'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float',
         'BalanceGross': 'float', 'ChgOffPrinGr': 'float', 'GrAppv': 'float',
         'SBA_Appv': 'float'})

    # Create a new column with the industry the NAICS code represents
    df_copy['Industry'] = df_copy['NAICS'].astype('str').apply(lambda x: x[:2])
    df_copy['Industry'] = df_copy['Industry'].map({
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
    df_copy.dropna(subset=['Industry'], inplace=True)

    # Create flag column IsFranchise based on FranchiseCode column
    df_copy.loc[(df_copy['FranchiseCode'] <= 1), 'IsFranchise'] = 0
    df_copy.loc[(df_copy['FranchiseCode'] > 1), 'IsFranchise'] = 1

    # Adjust current flag identifiers for NewExist, RevLineCr, LowDoc, and MIS_Status
    # NewExist
    # Make sure NewExist has only 1s and 2s; Remove records where NewExist isn't 1 or 2
    df_copy = df_copy[(df_copy['NewExist'] == 1) | (df_copy['NewExist'] == 2)]

    # NewExist - 0 = Existing business, 1 = New business; Renamed to NewBusiness
    df_copy.loc[(df_copy['NewExist'] == 1), 'NewBusiness'] = 0
    df_copy.loc[(df_copy['NewExist'] == 2), 'NewBusiness'] = 1

    # RevLineCr and LowDoc
    # Remove records where RevLineCr != 'Y' or 'N' and LowDoc != 'Y' or 'N'
    df_copy = df_copy[(df_copy['RevLineCr'] == 'Y') | (df_copy['RevLineCr'] == 'N')]
    df_copy = df_copy[(df_copy['LowDoc'] == 'Y') | (df_copy['LowDoc'] == 'N')]

    # RevLineCr and LowDoc - 0 = No, 1 = Yes
    df_copy['RevLineCr'] = np.where(df_copy['RevLineCr'] == 'N', 0, 1)
    df_copy['LowDoc'] = np.where(df_copy['LowDoc'] == 'N', 0, 1)

    # Make Default target field based on MIS_Status so P I F = 1 and CHGOFF = 0
    # Allows us to see what features are prevalent in a defaulted loan
    df_copy['Default'] = np.where(df_copy['MIS_Status'] == 'P I F', 0, 1)

    # Convert ApprovalDate and DisbursementDate columns to datetime values
    # ChgOffDate not changed to datetime since it is not of value and will be removed later
    df_copy[['ApprovalDate', 'DisbursementDate']] = df_copy[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)

    # Create DisbursementFY field for time selection criteria later
    df_copy['DisbursementFY'] = df_copy['DisbursementDate'].map(lambda record: record.year)

    # Create DaysToDisbursement column which calculates the number of days passed between DisbursementDate and
    # ApprovalDate
    df_copy['DaysToDisbursement'] = df_copy['DisbursementDate'] - df_copy['ApprovalDate']

    # Change DaysToDisbursement from a timedelta64 dtype to an int64 dtype
    # Converts series to str, removes all characters after the space before 'd' in days for each record, then changes
    # the dtype to int
    df_copy['DaysToDisbursement'] = \
        df_copy['DaysToDisbursement'].astype('str').apply(lambda record: record[:record.index('d') - 1]).astype('int64')

    # Create StateSame flag field which identifies where the business State is the same as the BankState
    df_copy['StateSame'] = np.where(df_copy['State'] == df_copy['BankState'], 1, 0)

    # Create SBA_AppvPct field since the guaranteed amount is based on a percentage of the gross loan amount
    # rather than dollar amount in most situations
    df_copy['SBA_AppvPct'] = df_copy['SBA_Appv'] / df_copy['GrAppv']

    # Create AppvDisbursed flag field signifying if the loan amount disbursed was equal to the full amount approved
    df_copy['AppvDisbursed'] = np.where(df_copy['DisbursementGross'] == df_copy['GrAppv'], 1, 0)

    # Format dtypes where necessary after feature engineering
    df_copy = df_copy.astype({'RevLineCr': 'int64', 'LowDoc': 'int64', 'Default': 'int64', 'IsFranchise': 'int64',
                              'NewBusiness': 'int64', 'StateSame': 'int64', 'AppvDisbursed': 'int64'})

    # Remove unnecessary columns
    df_copy.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist',
                          'FranchiseCode', 'ChgOffDate', 'NewExist', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr',
                          'SBA_Appv', 'MIS_Status'], inplace=True)

    # Field for loans backed by Real Estate (loans with a term of at least 20 years)
    df_copy['RealEstate'] = np.where(df_copy['Term'] >= 240, 1, 0)

    # Field for loans active during the Great Recession (2007-2009)
    df_copy['GreatRecession'] = np.where(
        ((2007 <= df_copy['DisbursementFY']) & (df_copy['DisbursementFY'] <= 2009)) |
        ((df_copy['DisbursementFY'] < 2007) & (df_copy['DisbursementFY'] + (df_copy['Term']/12) >= 2007)), 1, 0)

    # Only look at records with a DisbursementFY through 2010
    df_copy = df_copy[df_copy['DisbursementFY'] <= 2010]

    # Create flag to signify if a larger amount was disbursed than what the Bank had approved
    df_copy['DisbursedGreaterAppv'] = np.where(df_copy['DisbursementGross'] > df_copy['GrAppv'], 1, 0)

    # Remove records with loans disbursed prior to being approved
    df_copy = df_copy[df_copy['DaysToDisbursement'] >= 0]

    # Correlation Matrix
    corr_matrix(df=df_copy)

    # Exploring with Visualizations
    # Total/Average disbursed loan amount by industry
    # Create a groupby object on Industry for use in visualization
    industry_group = df_copy.groupby(['Industry'])

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
    stacked_setup(df=df_copy, col='Industry', axes=ax1a)
    ax1a.set_xticklabels(df_copy.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                         rotation=35, horizontalalignment='right', fontsize=6)

    ax1a.set_title('Number of PIF/Defaulted Loans by Industry')
    ax1a.set_xlabel('Industry')
    ax1a.set_ylabel('Number of PIF/Defaulted Loans')
    ax1a.legend()

    # Number of PIF and Defaulted loans by State
    stacked_setup(df=df_copy, col='State', axes=ax2a)

    ax2a.set_title('Number of PIF/Defaulted Loans by State')
    ax2a.set_xlabel('State')
    ax2a.set_ylabel('Number of PIF/Defaulted Loans')
    ax2a.legend()

    plt.tight_layout()
    plt.show()

    # Paid in full and Defaulted loans by DisbursementFY
    fig4, ax4 = plt.subplots(figsize=(15, 5))

    stack_data = df_copy.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')
    x = stack_data.index
    y = [stack_data[1], stack_data[0]]

    ax4.stackplot(x, y, labels=['Default', 'Paid in full'])
    ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
    ax4.set_xlabel('Disbursement Year')
    ax4.set_ylabel('Number of PIF/Defaulted Loans')
    ax4.legend(loc='upper left')

    plt.show()

    # Paid in full and defaulted loans backed by Real Estate
    fig5 = plt.figure(figsize=(20, 10))

    ax1b = fig5.add_subplot(1, 2, 1)
    ax2b = fig5.add_subplot(1, 2, 2)

    stacked_setup(df=df_copy, col='RealEstate', axes=ax1b)
    ax1b.set_xticks(df_copy.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default').index)
    ax1b.set_xticklabels(labels=['No', 'Yes'])

    ax1b.set_title('Number of PIF/Defaulted Loans backed by Real Estate from 1984-2010', fontsize=15)
    ax1b.set_xlabel('Loan Backed by Real Estate')
    ax1b.set_ylabel('Number of Loans')
    ax1b.legend()

    # Paid in full and defaulted loans active during the Great Recession
    stacked_setup(df=df_copy, col='GreatRecession', axes=ax2b)
    ax2b.set_xticks(df_copy.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default').index)
    ax2b.set_xticklabels(labels=['No', 'Yes'])

    ax2b.set_title('Number of PIF/Defaulted Loans Active during the Great Recession from 1984-2010', fontsize=15)
    ax2b.set_xlabel('Loan Active during Great Recession')
    ax2b.set_ylabel('Number of Loans')
    ax2b.legend()

    plt.show()

    # Modeling
    # One-hot encode categorical data
    df_copy = pd.get_dummies(df_copy)

    # Establish target and feature fields
    y = df_copy['Default']
    X = df_copy.drop('Default', axis=1)

    # Scale the feature values prior to modeling
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)

    # Split into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)

    # Initialize models
    log_reg = LogisticRegression(random_state=2)
    xgboost = XGBClassifier(random_state=2)

    # Modeling operations - Fit, predictions, print model performance metrics
    # Logistic Regression
    print("Logistic Regression esults:")
    modeling(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, model=log_reg)

    # XGBoost Classifier
    print("XGBoost Classifier results:")
    modeling(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, model=xgboost)


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

def modeling(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series, model):
    """
    Creates a pipeline that performs feature selection to select the 10 most important features, fits a specified model
    to the training data, makes a prediction, and prints a classification report with the results
    :param x_train: Features used to train the model
    :param y_train: Target used to train the model
    :param x_val: Validation set of features used for predictions
    :param y_val: Validation set of Target used for calculating model performance
    :param model: Model object used for modeling - the type of model you'd like to use
    :return: Classification report showing Precision, Recall, F1-Score, and accuracy for evaluating model performance
    """

    # Build pipeline for feature selection and modeling; SelectKBest defaults to top 10 important features
    pipe = Pipeline(steps=[
        ('feature_selection', SelectKBest()),
        ('model', model)
    ])

    # Train te model and make predictions
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_val)

    # Print the results
    print(classification_report(y_val, y_pred, digits=3))


if __name__ == '__main__':
    main()
