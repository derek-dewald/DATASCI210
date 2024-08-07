# Uniswap Project

## Team Google Drive
https://drive.google.com/drive/u/0/folders/1lBbvKi-utrPpTvPr63MUHXRLKcBlyHoK

## Team Website
mydiamondhands.io

## Website Git Repo
https://github.com/scaperothian/diamondhands-demo

## Streamlit App Git Repo
https://github.com/shirshikartejas/arbitrage_playgroud_v2/tree/main/models

## Table of Contents for Current Git Repo
Overview Summary of what is located in the Github Repo and Where it is located.

## Code
TheGraph_Transaction_Extraction.ipynb : Original Request for Transaction Data, as extracted from TheGraph. TheGraph underwent material updgrade between time of initial extraction and Jul-24, which resulted in this query no longer being functional.

Supplemental_InfoRequest.ipynb: Supplemental File, used in conjunction with a particular Transaction ID, to extract additional information, such as Max Gas Offered, Noonce and additional insight into Transaction ID.

Etherscan_Data_Extract_API.ipynb : Transaction Level Extraction from Etherscan to replace real-time (or near real time extraction), to replace Broken Graph call (TheGraph_Transaction_Extraction.ipynb)
Etherscan_Data_Extract_Jul18_V2.ipynb: Second iteration of Etherscan_Data_Extract (Parameters change, slight difference to timing logic).

Generate Dataset.ipynb: Function utilized to build upon foundational extract dataset and create Activity Level Metrics and references (Need to point to raw data to work correctly, currently pointing at specific location).

TRADER_DATASET - Function utilizing final dataset( Originally created manually, as final dataset not centrally posted at time of creation). Utilized to generate User Level Information, Metrics and Datasets. User datasets are not published directly due to duplication and size, need to utilzie this funciton to recreate. 

DATASCI210_DATA_VISUALIZATION_FUNCTIONS.py: Functions utilized in the creation of Trader Level Visualizations.
DATASCI210_Functions.py: Functions utilized in creation of Dataset. Includes derivation of all variables in Trader Dataet and Generate Dataset.



## Datasets
ALL_COLUMN_STATISTICAL_REVIEW: Review of Numberic Columns statistical average, Decile and differences, measured as a total population, only instances where Arbitrage is Present and where Arbitrage is Not Present. 
WBTC_Market_Price: Historical Market price of WBTC (Used in preliminary exploration).
WETH_Market_Price: Historical Market Price of WETH (Used in preliminary exploration).
hourly_XXXX. Files representing initial raw extraction hourly (used in exloration in advance of final dataset).
weth_to_usd_hist.xlsx/ weth_to_usdc_hist.xlsx - Historical representations of data.

Due to Size contraints, can not be easily stored in Github.
Final Dataset:
https://drive.google.com/drive/u/0/folders/1lBbvKi-utrPpTvPr63MUHXRLKcBlyHoK/arbitrage_dataset_20230611_20240613.zip (663.1MB)

Supplemental:
Supplemental File including Additional Gas Information, sourced by Transaction ID
https://drive.google.com/drive/u/0/folders/1lBbvKi-utrPpTvPr63MUHXRLKcBlyHoK/All_Supplemetal_Gas_Columns.csv (442.5MB)

Supplemental File including Additional Transaction Remitter information (File Missing Records, due to unknown API Extraction error)
https://drive.google.com/drive/u/0/folders/1lBbvKi-utrPpTvPr63MUHXRLKcBlyHoK/INITIAL_SUPPLEMENTAL_DATASET.csv (982.3MB)

Supplemental File including Additional Transaction Remitter information (Included transactions missed in initial Supplemental review)
https://drive.google.com/drive/u/0/folders/1lBbvKi-utrPpTvPr63MUHXRLKcBlyHoK/FINAL_SUPPLEMENTAL_DATASET.csv (432.90MB)

## Data Dictionary
Excel File of Data Dictionary for Final Dataset (with extention as available in Generate Dataset.ipynb

## EDA_Files
Extraction of All Numeric Variables identified as potentially revelant in Final Dataset, with Extention from Generate Dataset.ipynb (Data Dictionary Column Data Review (Binary Flag to include).

## [Detailed Project Plan](https://docs.google.com/document/d/1Oqdw755-bDg8vmPXqo-o7OxiThfNmEx8LH00WVWDaoU/edit?usp=sharing)<br>
Project information, value propositions, etc.

## [Project Schedule](https://docs.google.com/spreadsheets/d/1VNki3n7ZFB3otNgFqgAbgRNCWH6QF1p19KyGPDTUk-0/edit?usp=sharing)<br>
Estimated weekly milestones.
<br>
<br>
## Weekly Updates
[Week 4 Update](https://docs.google.com/presentation/d/1I4c74nZAT0lLQ7PlLA_nYih1yGYSzrGttBdHNvVxqa8/edit?usp=sharing) (30-May-24) - Still baselining on the objectives, some basics on terminology.<br>
[Week 5 Update](https://docs.google.com/presentation/d/1zGBGNFSkEttiE3OEr7CQOEIr8wSVPhynn23S_Lwo5uY/edit?usp=sharing) (06-Jun-24) - Presentation 1: Arbitrage details including research references, EDA, etc.<br>
[Week 6 Update](https://docs.google.com/presentation/d/1pYRLEQKx76R_Qkabx9OGdzJ1OIg-eKARIyjMd-pJwvg/edit?usp=sharing) (13-Jun-24) - Clear objective, tutorial refinement, EDA including dataset of swaps (2M transactions).<br>
[Week 7 Update](https://docs.google.com/presentation/d/1-0u_ZmkhvCD1P24fWFkwsIZuPIBl_Wyn5EOvXjLAs20/edit?usp=sharing) (20-Jun-24) - Updates and reordering slides<br>
[Week 8 Update](https://docs.google.com/document/d/1kJqzaGfBTkTcZLSdInZIGjZeftoowl_GlgQJiU64Jd8/edit?usp=sharing) (27-Jun-24) - Project Pitch
[Week 10 Update](https://docs.google.com/presentation/d/1Oeb7TE90p9eSLa8IhEUxMtKpo6s8t_8bPqAgZKBd7MA/edit?usp=sharing) (11-Jul-24) - Presentation 2
