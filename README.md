# SBA_LoanApproval
Investigating whether an SBA loan should be approved or not based on historical SBA loan data from 1987 to 2014.

This repository will be used to store my work and track my progress through a personal data science project to apply what I've learned through various resources, including proper use of Github and modeling techniques.

## Why SBA Loan Approval?
My first job out of college was at a community bank as a Credit Analyst, where I spent my time underwriting loans for small businesses. I loved being able to dig into the financial statements for each business and see how businesses in different industries operate. This role taught me the importance of small businesses and the role they play in our communities. It also taught me about some of the struggles entreprenuers face when starting a business, including the initial capital necessary to get started.

Small business owners often seek out SBA (Small Business Association) loans because they guarantee part of the loan. Without going into too much detail, this basically means that the SBA will cover some of the losses should the business default on the loan, which lowers the risk involved for the business owner(s). This increases the risk to the SBA however, which can sometimes make it difficult to get accepted for one of their loan programs. The SBA will be particularly important now with the COVID-19 pandemic which is impacting many small businesses around the world, crippling most of them. I thought it would be interesting to see I could determine whether or not an SBA loan should be accepted or not given certain characteristics like the industry of the business, the size of the loan, amount of the loan that is guaranteed, etc.

## About the Data
The information below was provided by Mirbek Toktogaraev on Kaggle, under the dataset titled "Should This Loan be Approved or Denied?".

### Context
The dataset is from the U.S. Small Business Administration (SBA)

The U.S. SBA was founded in 1953 on the principle of promoting and assisting small enterprises in the U.S. credit market (SBA Overview and History, US Small Business Administration (2015)). Small businesses have been a primary source of job creation in the United States; therefore, fostering small business formation and growth has social benefits by creating job opportunities and reducing unemployment.

There have been many success stories of start-ups receiving SBA loan guarantees such as FedEx and Apple Computer. However, there have also been stories of small businesses and/or start-ups that have defaulted on their SBA-guaranteed loans.

### Content
Shape of the data: 899164 rows and 27 columns

### Data Dictionary
| Variable Name	| Description |
| ------------- | ----------- |
| LoanNr_ChkDgt	| Identifier Primary key |
| Name	| Borrower name |
| City	| Borrower city |
| State	| Borrower state |
| Zip	| Borrower zip code |
| Bank | Bank name |
| BankState	| Bank state |
| NAICS	| North American industry classification system code |
| ApprovalDate	| Date SBA commitment issued |
| ApprovalFY	| Fiscal year of commitment |
| Term	| Loan term in months |
| NoEmp	| Number of business employees |
| NewExist	| 1 = Existing business, 2 = New business |
| CreateJob	| Number of jobs created |
| RetainedJob |	Number of jobs retained |
| FranchiseCode	| Franchise code, (00000 or 00001) = No franchise |
| UrbanRural	| 1 = Urban, 2 = rural, 0 = undefined |
| RevLineCr	| Revolving line of credit: Y = Yes, N = No |
| LowDoc	| LowDoc Loan Program: Y = Yes, N = No |
| ChgOffDate	| The date when a loan is declared to be in default |
| DisbursementDate	| Disbursement date |
| DisbursementGross	| Amount disbursed |
| BalanceGross	| Gross amount outstanding |
| MIS_Status	| Loan status charged off = CHGOFF, Paid in full =PIF |
| ChgOffPrinGr	| Charged-off amount |
| GrAppv | Gross amount of loan approved by bank |
| SBA_Appv	| SBA’s guaranteed amount of approved loan |

Description of the first two digits of NAICS. 

| Sector	| Description |
| ------- | ----------- |
| 11	| Agriculture, forestry, fishing and hunting |
| 21	| Mining, quarrying, and oil and gas extraction |
| 22	| Utilities |
| 23	| Construction |
| 31–33	| Manufacturing |
| 42	| Wholesale trade |
| 44–45	| Retail trade |
| 48–49	| Transportation and warehousing |
| 51	| Information |
| 52	| Finance and insurance |
| 53	| Real estate and rental and leasing |
| 54	| Professional, scientific, and technical services |
| 55	| Management of companies and enterprises |
| 56	| Administrative and support and waste management and remediation services |
| 61	| Educational services |
| 62	| Health care and social assistance |
| 71	| Arts, entertainment, and recreation |
| 72	| Accommodation and food services |
| 81	| Other services (except public administration) 92 Public administration |

## Acknowledgements
Data was posted to Kaggle by Mirbek Toktogaraev, with the original data set id from “Should This Loan be Approved or Denied?”: A Large Dataset with Class Assignment Guidelines.
by: Min Li, Amy Mickel & Stanley Taylor

To link to this article: https://doi.org/10.1080/10691898.2018.1434342

