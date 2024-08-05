import pandas as pd
import numpy as np
import os
import datetime
import math

def ImportFiles(text,location=''):
    '''
    Function to iterate through a Windows folder and read Series of CSV files. Folder can Only contain CSV files, not xlsx.
    If need can build another exclusion compoennt in temp_list, not currently needed

    arg
        text: String which file should include
        location: If blank takes from current working directory, if contained else can explicitly define path

    example
        df = ImportFiles('arbi','Arbitrage_Files')

    '''
    if len(location)==0:
        temp_list = [x for x in os.listdir() if x.find(text)!=-1]
    else:
        temp_list = [f"{location}//{x}" for x in os.listdir(location) if x.find(text)!=-1]
    
    final_df = pd.DataFrame()
    
    for report in temp_list:
        final_df = pd.concat([final_df,pd.read_csv(report)])
    
    return final_df

def create_segments_from_dataframe_column(df,
                                          column_name,
                                          segments=10,
                                          exclude_blanks=1,
                                          exclude_zeros=1,
                                          output_item='value_df',
                                          segment_column_name='Segment'):

    if segments <2: 
        return print('Requires a Minimum of 2 Segments, Recommend no less than 3')
    
    temp_df = df.copy()
    if exclude_blanks==1:
        blanks_removed = len(temp_df[temp_df[column_name].isnull()])
        temp_df = temp_df[temp_df[column_name].notnull()]
    
    if exclude_zeros==1:
        zeros_removed = len(temp_df[temp_df[column_name]==0])
        temp_df = temp_df[temp_df[column_name]!=0]
    
    column_list = temp_df[column_name].tolist()
    column_list.sort()
    length_of_df = len(column_list)
    break_point = math.ceil(length_of_df/segments)
    if segments>=length_of_df:
        return print(f"Sample Size Insufficient to Warrant Calculation for column {column_name}, please review data")

    record_position = list(range(0,length_of_df,break_point))
    record_value = [column_list[x] for x in record_position]

    if output_item=='value_df':
        return pd.DataFrame(record_value,index=[f'Segment {x+1}' for x in range(len(record_value))],columns=[column_name]).T
    elif output_item=='record_value':
        return record_value
    elif output_item=='record_position':
        return record_position
    elif output_item =='segment_if_df_column':
        print(f"Segments:{reocrd_value}")
        new_name = f"Segment {column_name}"
        df[new_name] = df[column_name].apply(lambda x:find_position(x,record_value))
        return df

def extract_df(pool_id):
  """
  Funtion to extract transaction level files from Folder, iterates through all files looking for combination pool_id_XXXXXX

  Location Updated to work in Google Drive.

  """
  final_df = pd.DataFrame()

  for i in [x for x in os.listdir() if x.find(f'pool_id_{pool_id}')!=-1]:
    temp_df = pd.read_csv(f"{i}")
    try:
      temp_df['time']
      pass
    except:
      temp_df['DATE'] = temp_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).replace(hour=0, minute=0, second=0, microsecond=0))
    final_df = pd.concat([final_df,temp_df])
  final_df['amountUSD'] = final_df['amountUSD'].apply(lambda x:float(x))

  try:
    final_df['DATE'] = final_df['time'].apply(lambda x: datetime.datetime(int(x[:4]),int(x[5:7]),int(x[8:10])))
  except:
    pass
  try:
    return final_df.drop('Unnamed: 0',axis=1).reset_index(drop=True)
  except:
    return final_df.reset_index(drop=True)


def PoolActivitybyUser(df,
                       sender_recipient,
                       segments=10,
                       show_visuals=1,
                       brackets=['Novice','Amateur','Active','Trader','Veteran']):

    df['COUNT'] = 1

    # Create individual groupby 
    temp_df = df[['COUNT',sender_recipient]].groupby(sender_recipient).sum().reset_index().rename(columns={sender_recipient:"USER_ID"}).sort_values('COUNT')
    temp_df['CUMMULATIVE_SUM'] = temp_df['COUNT'].cumsum()
    temp_df['CUMMULATIVE_PERC'] = temp_df['CUMMULATIVE_SUM']/temp_df['COUNT'].sum()
    temp_df['OBSERVATION'] = 1
    temp_df['OBSERVATION'] = temp_df['OBSERVATION'].cumsum()
    
    return temp_df

def TestHistogram(pd_series):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
    filtered_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]
    log_data = np.log1p(data)  # log1p is used to avoid log(0)
    winsorized_data = winsorize(data, limits=[0.05, 0.05])
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(data, bins=10, color='blue', edgecolor='black')
    plt.title('Original Data')
    plt.subplot(2, 2, 2)
    plt.hist(filtered_data, bins=10, color='green', edgecolor='black')
    plt.title('Filtered Data (No Outliers)')
    plt.subplot(2, 2, 3)
    plt.hist(log_data, bins=10, color='purple', edgecolor='black')
    plt.title('Log Transformed Data')
    plt.subplot(2, 2, 4)
    plt.hist(winsorized_data, bins=10, color='orange', edgecolor='black')
    plt.title('Winsorized Data')
    
    plt.tight_layout()
    plt.show()

def CreateActivityByDATE(df,
                         sender,
                         recipient,
                         value,
                         return_value='summary',pool_name=""):

    temp_df = df.copy()
    buys = temp_df[[sender,'DATE',value]].rename(columns={sender:'USER_ID'})
    buys['TRADE'] = 'BUY'
    sells = temp_df[[recipient,'DATE',value]].rename(columns={recipient:'USER_ID'})
    sells['TRADE'] = 'SELL'
    final_df = pd.concat([buys,sells])
    final_df.sort_values('DATE',inplace=True)
    final_df['FIRST_TRADE'] =  (~final_df['USER_ID'].duplicated()).astype(int)
    final_df['TOTAL_TRADES'] =  1
    if pool_name !="":
        final_df['POOL_NAME'] = pool_name
    if return_value=='summary':
        return final_df[['FIRST_TRADE','TOTAL_TRADES','DATE']].groupby('DATE').sum().reset_index()
    else:
        return final_df

def fetch_transaction_details(tx_id,
                              api_key,
                              etherscan_url = 'https://api.etherscan.io/api',
                              rate_limit=5):
    params = {
        'module': 'proxy',
        'action': 'eth_getTransactionByHash',
        'txhash': tx_id,
        'apikey': api_key
    }
        
    response = requests.get(etherscan_url, params=params)
    data = response.json()

    try:
        if 'input' in data['result']:
            del data['result']['input']
        return transaction_to_dataframe(data['result'])
    except:
        pass

def hex_to_int(hex_str):
    try:
        return int(hex_str, 16) if hex_str.startswith('0x') else hex_str
    except ValueError:
        return hex_str

# Function to convert transaction data to DataFrame
def transaction_to_dataframe(transaction_data):
    # Convert hex values to integers where applicable, but keep large numbers as strings
    for key, value in transaction_data.items():
        if isinstance(value, str) and value.startswith('0x'):
            if key in ['blockNumber', 'gas', 'nonce', 'transactionIndex', 'chainId']:
                transaction_data[key] = hex_to_int(value)
            else:
                # Keep the value as string if it's too large to handle as an integer
                try:
                    transaction_data[key] = hex_to_int(value)
                except OverflowError:
                    pass

    # Create a DataFrame from the dictionary
    df = pd.DataFrame([transaction_data])
    
    return df

def CreateStandardizedDefinitions(df):
    '''

    '''
    temp_df = df.copy()
    # Remove Instances Where there is not 

    temp_df['BUYER_POOL_1'] = np.where(temp_df['to_trade1'].notnull(),temp_df['to_trade1'],temp_df['p1.recipient'])
    temp_df['SELLER_POOL_1'] = np.where(temp_df['from_trade1'].notnull(),temp_df['from_trade2'],temp_df['p1.sender'])
    temp_df['BUYER_POOL_2'] = np.where(temp_df['to_trade1'].notnull(),temp_df['to_trade2'],temp_df['p0.recipient'])
    temp_df['SELLER_POOL_2'] = np.where(temp_df['from_trade2'].notnull(),temp_df['from_trade2'],temp_df['p0.sender'])

    temp_df['BUYER_POOL_1_COUNT_IN_DATASET'] = temp_df.groupby('BUYER_POOL_1').cumcount() + 1
    temp_df['SELLER_POOL_1_COUNT_IN_DATASET'] = temp_df.groupby('SELLER_POOL_1').cumcount() + 1
    temp_df['BUYER_POOL_2_COUNT_IN_DATASET'] = temp_df.groupby('BUYER_POOL_2').cumcount() + 1
    temp_df['SELLER_POOL_2_COUNT_IN_DATASET'] = temp_df.groupby('SELLER_POOL_2').cumcount() + 1

    return temp_df


api_keys = ['Y8YP3J3BTG1DRZZPQ3F2I41E88I9GSUXNU',
            'YED1TTBKJJHVXWNNK8XTAHJDY9H9H4HFUH',
            'JUQBES238PMPJKI2EJWAJJ2V8973EMCEZH',
            'FYQ8DAUFF2RXE4JH8PX7IXFCGVAPQEX4ZU',
            '6I496I2Z7SVI6MFP1VFWD2B951RB7AZDIS']

def SummarizeAllTransactions(pool_a,
                             pool_b,
                             trade_val = 't0_amount_abs'):
    '''
    How are we going to interpret Buying and Selling
    
    
    '''

    a = CreateActivityByDATE(pool_a,
                             sender='SELLER_POOL_1',
                             recipient='BUYER_POOL_1',
                             value=f'p1.{trade_val}',
                             return_value='',
                             pool_name="POOL_A")
    a.rename(columns={f'p1.{trade_val}':'TRADE_VAL'},inplace=True)

    b = CreateActivityByDATE(pool_b,
                             sender='SELLER_POOL_2',
                             recipient='BUYER_POOL_2',
                             value=f'p0.{trade_val}',
                             return_value='',
                             pool_name="POOL_B")
    b.rename(columns={f'p0.{trade_val}':'TRADE_VAL'},inplace=True)
    
    total_trans_df = pd.concat([a,b]).drop(['FIRST_TRADE','TOTAL_TRADES'],axis=1)
    total_trans_df['RECORD_COUNT'] = 1
    total_trans_df['TRADE_VAL'] = total_trans_df['TRADE_VAL'].apply(lambda x:float(x))

    # Generate Summary Table
    temp_summ = total_trans_df[['USER_ID','TRADE','POOL_NAME','RECORD_COUNT','TRADE_VAL']].groupby(['USER_ID','TRADE','POOL_NAME']).sum()

    user_summary_df = temp_summ.pivot_table(
        index='USER_ID',
        columns=['TRADE', 'POOL_NAME'],
        values={'RECORD_COUNT': 'sum', 'TRADE_VAL': 'sum'},
        aggfunc='sum',
        fill_value=0)

    user_summary_df = user_summary_df.reset_index()

# Flatten the multi-level columns
    user_summary_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in user_summary_df.columns]
    user_summary_df.rename(columns={'USER_ID__':'USER_ID'},inplace=True)
    user_summary_df['BOUGHT_POOL_A'] = (user_summary_df[('RECORD_COUNT_BUY_POOL_A')] > 0).astype(int)
    user_summary_df['SOLD_POOL_A'] = (user_summary_df[('RECORD_COUNT_SELL_POOL_A')] > 0).astype(int)
    user_summary_df['BOUGHT_POOL_B'] = (user_summary_df[('RECORD_COUNT_BUY_POOL_B')] > 0).astype(int)
    user_summary_df['SOLD_POOL_B'] = (user_summary_df[('RECORD_COUNT_SELL_POOL_B')] > 0).astype(int)
    user_summary_df['BOUGHT_SOLD_POOL_A'] = np.where((user_summary_df['BOUGHT_POOL_A']>0)&(user_summary_df['SOLD_POOL_A']>0),1,0)
    user_summary_df['BOUGHT_SOLD_POOL_B'] = np.where((user_summary_df['BOUGHT_POOL_B']>0)&(user_summary_df['SOLD_POOL_B']>0),1,0)
        
    user_summary_df['TRADED_POOL_A'] =     np.where((user_summary_df['BOUGHT_POOL_A']>0)|(user_summary_df['SOLD_POOL_A']>0),1,0)
    user_summary_df['TRADED_POOL_B'] =     np.where((user_summary_df['BOUGHT_POOL_B']>0)|(user_summary_df['SOLD_POOL_B']>0),1,0)

    user_summary_df['TRADED_BOTH_POOLS'] = np.where((user_summary_df['TRADED_POOL_A']>0)&(user_summary_df['TRADED_POOL_B']>0),1,0)
    
    user_summary_df['ONLY_TRADED_POOL_A'] =     np.where((user_summary_df['TRADED_POOL_A']>0)&(user_summary_df['TRADED_POOL_B']==0),1,0)
    user_summary_df['ONLY_TRADED_POOL_B'] =     np.where((user_summary_df['TRADED_POOL_A']==0)&(user_summary_df['TRADED_POOL_B']>0),1,0)
            
    user_summary_df['ONLY_PURCHASED'] =    np.where((user_summary_df['SOLD_POOL_A']==0)&(user_summary_df['SOLD_POOL_B']==0),1,0)
    user_summary_df['ONLY_SOLD'] =         np.where((user_summary_df['BOUGHT_POOL_A']==0)&(user_summary_df['BOUGHT_POOL_B']==0),1,0)
    
    user_summary_df['TOTAL_PURCHASE_VOL'] =      user_summary_df['RECORD_COUNT_BUY_POOL_A'] + user_summary_df['RECORD_COUNT_BUY_POOL_B']
    user_summary_df['TOTAL_SELL_VOL'] =          user_summary_df['RECORD_COUNT_SELL_POOL_A'] + user_summary_df['RECORD_COUNT_SELL_POOL_B']
    
    user_summary_df['TOTAL_ACTIVITY_VOL_POOL_A'] =      user_summary_df['RECORD_COUNT_BUY_POOL_A'] + user_summary_df['RECORD_COUNT_SELL_POOL_A']
    user_summary_df['TOTAL_ACTIVITY_VOL_POOL_B'] =      user_summary_df['RECORD_COUNT_BUY_POOL_B'] + user_summary_df['RECORD_COUNT_SELL_POOL_B']
    user_summary_df['TOTAL_ACTIVITY_VOL'] =             user_summary_df['TOTAL_ACTIVITY_VOL_POOL_A'] + user_summary_df['TOTAL_ACTIVITY_VOL_POOL_B']

    user_summary_df['TOTAL_PURCHASE_VAL'] =      user_summary_df['TRADE_VAL_BUY_POOL_A'] + user_summary_df['TRADE_VAL_BUY_POOL_B']
    user_summary_df['TOTAL_SELL_VAL'] =          user_summary_df['TRADE_VAL_SELL_POOL_A'] + user_summary_df['TRADE_VAL_SELL_POOL_B']
    
    user_summary_df['TOTAL_ACTIVITY_VAL_POOL_A'] =      user_summary_df['TRADE_VAL_BUY_POOL_A'] + user_summary_df['TRADE_VAL_SELL_POOL_A']
    user_summary_df['TOTAL_ACTIVITY_VAL_POOL_B'] =      user_summary_df['TRADE_VAL_BUY_POOL_B'] + user_summary_df['TRADE_VAL_SELL_POOL_B']
    user_summary_df['TOTAL_ACTIVITY_VAL'] =             user_summary_df['TOTAL_ACTIVITY_VAL_POOL_A'] + user_summary_df['TOTAL_ACTIVITY_VAL_POOL_B']

    user_summary_df['NET_POSITION_CHANGE'] =      user_summary_df['TOTAL_PURCHASE_VAL'] - user_summary_df['TOTAL_SELL_VAL']
        
    info_dict = {'Total Traders':len(user_summary_df),
                 'Total Traders Buying From Pool A': user_summary_df['BOUGHT_POOL_A'].sum(),
                 'Total Traders Selling into Pool A': user_summary_df['SOLD_POOL_A'].sum(),
                 'Total Traders Buying From Pool B': user_summary_df['BOUGHT_POOL_B'].sum(),
                 'Total Traders Selling into Pool B': user_summary_df['SOLD_POOL_B'].sum(),
                 'Total Traders Buying and Selling Pool A': user_summary_df['BOUGHT_SOLD_POOL_A'].sum(),
                 'Total Traders Buying and Selling Pool B': user_summary_df['BOUGHT_SOLD_POOL_B'].sum(),
                 'Total Traders Transacting in Only Pool A': user_summary_df['ONLY_TRADED_POOL_A'].sum(),
                 'Total Traders Transacting in Only Pool B': user_summary_df['ONLY_TRADED_POOL_B'].sum(),
                 'Total Traders Transacting in Both Pools': user_summary_df['TRADED_BOTH_POOLS'].sum(),
                 'Total Traders Only Buying': user_summary_df['ONLY_PURCHASED'].sum(),
                 'Total Traders Only Sellilng': user_summary_df['ONLY_SOLD'].sum()
                }

    user_summary_stats = pd.DataFrame([info_dict.values()],columns=info_dict.keys(),index=['Summary Statistics']).T
    user_summary_stats['PERCENTAGE'] = [user_summary_stats.loc[x].item()/user_summary_stats.loc['Total Traders'].item() for x in user_summary_stats.index]

    return total_trans_df,user_summary_df,user_summary_stats.reset_index().rename(columns={'index':'Metric'})


def GetHistoricalTradeActivity(df,
                               merge_df="",
                               time='time',
                               new_column_name= 'ExchangeTradeCount',
                               nonce_seller_pool1= 'nonce_trade1',
                               seller1_column_name= 'SELLER_POOL_1',
                               nonce_seller_pool2='nonce_trade2',
                               seller2_column_name= 'SELLER_POOL_2'):
    '''
    Using Field Nonce Where available.

    Args:
        Takes a DataFrame 
    
    '''
    final_df = pd.concat([df[df[nonce_seller_pool1].notnull()][[seller1_column_name,nonce_seller_pool1]].rename(columns={seller1_column_name:'USER_ID',nonce_seller_pool1:new_column_name}),
                         df[df[nonce_seller_pool2].notnull()][[seller2_column_name,nonce_seller_pool2]].rename(columns={seller2_column_name:'USER_ID',nonce_seller_pool2:new_column_name})])

    final_df = final_df.sort_values(new_column_name,ascending=False).drop_duplicates('USER_ID')

    if len(merge_df)==0:
        return final_df
    else:
        final_df = merge_df.merge(final_df,on='USER_ID',how='left')
        final_df[new_column_name] = final_df[new_column_name].fillna(0)
        return final_df

def UserSegmentCreation(df,
                        vol_threshold,
                        val_threshold,
                        vol_column='USER_TOTAL_ACTIVITY',
                        val_column='TOTAL_ACTIVITY_VAL'):

    condition = [(df[vol_column]>=vol_threshold)&(df[val_column]>=val_threshold),
                 (df[vol_column]>=vol_threshold)&(df[val_column]<=val_threshold),
                 (df[vol_column]<=vol_threshold)&(df[val_column]>=val_threshold),
                 (df[vol_column]<=vol_threshold)&(df[val_column]<=val_threshold)]
    
    value = ['Investing Trader',
             'Trader',
             'Speculator',
             'Not Engaged']
    
    df['USER_SEGMENT'] = np.select(condition,value)

def CreateFinalDataSet(df,supplemental1,supplemental2):

    supplemental_df = pd.concat([supplemental1,supplemental2])
    
    # Merge On Transaction ID. Transaction ID was not explicitly saved in initial Data Extraction, as per the API Block Number was utilized.
    # However in attempting to merge, a number of Data Quality Issues were noted with blockNumber resulting in ~25% of data unable to merge
    # supplemented with missing transactions at Transaction ID. Will look to obtain all information at Transaction ID, but will take 
    # several days to replace.
    
    supp_by_TXN_ID = supplemental_df[supplemental_df['Transaction_ID'].notnull()]
    
    trade_1 = supp_by_TXN_ID.rename(columns={x:f'{x}_trade1' for x in supp_by_TXN_ID.columns if x.find('Transaction_ID')==-1})
    trade_1 = trade_1.rename(columns={'Transaction_ID':'p1.transaction_id'})
    trade_2 = supp_by_TXN_ID.rename(columns={x:f'{x}_trade2' for x in supp_by_TXN_ID.columns if x.find('Transaction_ID')==-1})
    trade_2 = trade_2.rename(columns={'Transaction_ID':'p0.transaction_id'})
    
    temp_df = df.merge(trade_1,on='p1.transaction_id',how='left').merge(trade_2,on='p0.transaction_id',how='left')
    
    # Having Merged Information which is available, need to break data into 2 pieces 1) Which represents transactions which have supplemental
    # Information a second which does not. 
    
    missing_info = temp_df[temp_df['blockHash_trade1'].isnull()].copy()
    final_Transaction_ID_MATCH = temp_df[temp_df['blockHash_trade1'].notnull()].copy()
    print(f"To Number of Transactions Matching on Transaction ID {len(final_Transaction_ID_MATCH)}")
    print(f"Number of Transaction Missing Link of Transaction ID {len(missing_info)}")
    
    # Need to DROP COLUMNS, which are all blank as I we will concat them in after filling
    supp_col = ['blockHash_trade1', 'blockNumber_trade1','from_trade1', 'gas_trade1', 'gasPrice_trade1','maxFeePerGas_trade1', 'maxPriorityFeePerGas_trade1',
                'hash_trade1', 'nonce_trade1', 'to_trade1','transactionIndex_trade1', 'value_trade1', 'type_trade1','accessList_trade1', 'chainId_trade1', 
                'v_trade1', 'r_trade1','s_trade1', 'yParity_trade1', 'blockHash_trade2','blockNumber_trade2', 'from_trade2', 'gas_trade2',
                'gasPrice_trade2', 'maxFeePerGas_trade2','maxPriorityFeePerGas_trade2', 'hash_trade2', 'nonce_trade2','to_trade2', 
                'transactionIndex_trade2', 'value_trade2','type_trade2', 'accessList_trade2', 'chainId_trade2', 'v_trade2','r_trade2', \
                's_trade2', 'yParity_trade2']
    
    missing_info.drop(supp_col,axis=1,inplace=True)
    
    # Replace Same Process above, for blockNumber
    
    # Iterate through BlockNumber
    supp_by_BlockNumber = supplemental_df[supplemental_df['Transaction_ID'].isnull()]
    
    # Fix Column Naming Conventions for clarity
    trade_1 = supp_by_BlockNumber.rename(columns={x:f'{x}_trade1' for x in supp_by_BlockNumber.columns if x.find('blockNumber')==-1})
    trade_1 = trade_1.rename(columns={'blockNumber':'p1.blockNumber'})
    trade_2 = supp_by_BlockNumber.rename(columns={x:f'{x}_trade2' for x in supp_by_BlockNumber.columns if x.find('blockNumber')==-1})
    trade_2 = trade_2.rename(columns={'blockNumber':'p0.blockNumber'})
    
    temp_df = missing_info.merge(trade_1,on='p1.blockNumber',how='left').merge(trade_2,on='p0.blockNumber',how='left')
    
    STILL_MISSING1 = temp_df[temp_df['blockHash_trade1'].isnull()].copy()
    STILL_MISSING2 = temp_df[temp_df['blockHash_trade2'].isnull()].copy()

    unique1 = STILL_MISSING1.drop_duplicates('p1.transaction_id')
    unique2 = STILL_MISSING2.drop_duplicates('p0.transaction_id')

    
    print(f"Numher of Transactions which are still Missing to Map to Transaction1: {len(STILL_MISSING1)}")
    print(f"Numher of Unique Transactions which are still Missing to Map to Transaction1: {len(unique1)}")
    
    print(f"Numher of Transactions which are still Missing to Map to Transaction2: {len(STILL_MISSING2)}")
    print(f"Numher of Unique Transactions which are still Missing to Map to Transaction1: {len(unique2)}")
    
    final_blockNumber_Match = temp_df[temp_df['blockHash_trade1'].notnull()].copy()
    
    return pd.concat([final_Transaction_ID_MATCH,final_blockNumber_Match])

def CalculateInterval(df,
                      column_name,
                      datetime_column,
                      minutes,
                      return_value='summary'):
    '''
    Function to take a particular column in a Datafarme and create intervals with statistical insights

    Args:
        df: Pandas Dataframe

        column_name: Name of Column (must be numeric)

        datetime_column: Column to group on, must be datetime format

        Minutes: Number of minutes to group by

        return_valu: User defined option:
            summary; returns a standalone dataframe
            merge: merges statistics back into dataframe

    '''

    temp = df.groupby(pd.Grouper(key=datetime_column,freq=f"{minutes}T"))
    temp1 = temp[column_name].agg(high='max',
                                  low='min',
                                  open='first',
                                  close='last',
                                  transactions='count',
                                  mean='mean',
                                  std_dev='std')

    if return_value=='summary':
        return temp1
    else:
        df['INTERVAL'] = df[datetime_column].dt.floor('5T')
        return pd.merge(df, temp1, left_on='INTERVAL', right_on=datetime_column, how='left')


def Supplemental_Dictionary():
    transaction_info = {
        "blockHash": (
            "Uniquely identifies the block containing the transaction.",
            "A unique string that represents the specific block in which the transaction is included."
        ),
        "blockNumber": (
            "Indicates the block's position in the blockchain.",
            "The sequential number of the block within the blockchain."
        ),
        "from": (
            "The sender's address, indicating who initiated the transaction.",
            "The Ethereum address of the account that sent the transaction."
        ),
        "gas": (
            "The maximum amount of gas units the transaction can consume.",
            "The upper limit of computational effort that the transaction is allowed to use."
        ),
        "gasPrice": (
            "The price per gas unit the sender is willing to pay.",
            "The amount of Ether the sender is willing to pay for each unit of gas."
        ),
        "maxFeePerGas": (
            "Maximum fee per gas the sender is willing to pay (specific to EIP-1559 transactions).",
            "The highest amount the sender will pay per unit of gas."
        ),
        "maxPriorityFeePerGas": (
            "Maximum priority fee per gas the sender is willing to pay to incentivize miners (specific to EIP-1559 transactions).",
            "An additional fee to prioritize the transaction."
        ),
        "hash": (
            "A unique identifier for the transaction.",
            "A unique string that identifies the transaction within the blockchain."
        ),
        "input": (
            "Encoded data sent with the transaction, often containing method calls and parameters for smart contract interactions.",
            "Data included in the transaction, typically used when calling functions in smart contracts."
        ),
        "nonce": (
            "The count of transactions sent from the sender's address.",
            "A counter that shows how many transactions the sender has made."
        ),
        "to": (
            "The recipient's address or contract address.",
            "The Ethereum address of the account or contract that receives the transaction."
        ),
        "transactionIndex": (
            "The position of the transaction within the block.",
            "The order of the transaction within its block."
        ),
        "value": (
            "The amount of Ether transferred (in Wei).",
            "The amount of Ether being sent in the transaction, measured in Wei (the smallest unit of Ether)."
        ),
        "type": (
            "The type of transaction, indicating if it's a legacy or EIP-1559 transaction.",
            "Specifies whether the transaction follows the old format or the new EIP-1559 format."
        ),
        "accessList": (
            "A list of addresses and storage keys that the transaction plans to access (specific to EIP-2930 transactions).",
            "Pre-defined list of addresses and storage slots the transaction will access, improving efficiency."
        ),
        "chainId": (
            "Identifies the Ethereum network.",
            "The ID of the network (mainnet, testnet, etc.) on which the transaction is executed."
        ),
        "v": (
            "Components of the ECDSA signature, used to verify the sender and the transaction's integrity.",
            "Part of the digital signature that verifies the transaction's authenticity."
        ),
        "r": (
            "Components of the ECDSA signature, used to verify the sender and the transaction's integrity.",
            "Part of the digital signature that verifies the transaction's authenticity."
        ),
        "s": (
            "Components of the ECDSA signature, used to verify the sender and the transaction's integrity.",
            "Part of the digital signature that verifies the transaction's authenticity."
        ),
        "yParity": (
            "Parity bit used for EIP-1559 transactions.",
            "A parameter that helps in transaction verification."
        )
    }
    
    return  pd.DataFrame.from_dict(transaction_info, orient='index', columns=['Description', 'Explanation'])
    
def SPLIT_DF_TO_CSV(df,
                    column,
                    records=10000):

    iterations = math.ceil(len(df)/records)
    print(iterations)

    for count in range(0,iterations):
        if count == 0:
            may_txnid[:records].to_csv('raw_Receipts_0.csv',index=False)
        else:
            current = records*count
            may_txnid[current:current+records].to_csv(f'raw_Receipts_{count}.csv',index=False)


def AggregatePoolCalculations(df,
                              merge_back=1,
                              trader='TRADER_POOL1',
                              trader1='TRADER_POOL2',
                              amount='p1.t0_amount_unique',
                              amount1='p0.t0_amount_unique',
                              vol_unique1 = 'BUY_SELL_BINARY1_unique',
                              vol_unique2 = 'BUY_SELL_BINARY2_unique',
                              date='DATE',
                              time='time'):
    
    # Need to Decouple Pool 1 and Pool 2 into distinct Time Sorted Values
    temp_0 = df[[date,time,trader,amount,vol_unique1]].rename(columns={trader:'TRADER',amount:'VALUE',vol_unique1:'UNI_COUNT'})
    temp_1 = df[[date,time,trader1,amount1,vol_unique2]].rename(columns={trader1:'TRADER',amount1:'VALUE',vol_unique2:'UNI_COUNT'})
    temp_ = pd.concat([temp_0,temp_1])
    
    # Add Count as we will consolidate in the case of Concurrent Timing Transactions
    # Add Absolute Value so we can track Aggregate Activity and Also Net Position.
    temp_['TRADE_COUNT'] = 1
    temp_['ABS_VALUE'] = temp_['VALUE'].apply(lambda x:abs(x))
    
    # Aggregate to ensure that transactions occuring at the exact same time are considered.
    
    temp_ = temp_.groupby([date,time,'TRADER']).sum().reset_index()
    temp_['TRADER_COMBINED_POOL_DAILY_VAL'] = temp_.groupby(['DATE','TRADER'])['ABS_VALUE'].cumsum()
    temp_['TRADER_COMBINED_POOL_DAILY_VOL'] = temp_.groupby(['DATE','TRADER'])['TRADE_COUNT'].cumsum()
    
    temp_['TRADER_COMBINED_POOL_NET_DAILY_VAL'] = temp_.groupby(['DATE','TRADER'])['VALUE'].cumsum()
    temp_['TRADER_NET_DAILY_BUY_SELL'] = temp_.groupby(['DATE','TRADER'])['UNI_COUNT'].cumsum()
    
    if merge_back==1:
        trade1_block = temp_.rename(columns={x:x.replace('TRADER',trader) for x in temp_.columns}).drop(['TRADE_COUNT','VALUE','ABS_VALUE','UNI_COUNT'],axis=1)
        trade2_block = temp_.rename(columns={x:x.replace('TRADER',trader1) for x in temp_.columns}).drop(['TRADE_COUNT','VALUE','ABS_VALUE','UNI_COUNT'],axis=1)

        return df.merge(trade1_block,on=["time",trader,'DATE'],how='left').merge(trade2_block,on=["time",trader1,'DATE'],how='left')
    return temp_

def TraderStatistics(df,
                     target,
                     merge_back=0,
                     trader='POOL1_TRADER',
                     trader1='POOL2_TRADER',
                     amount='p1.1.transaction_amt',
                     amount1='p1.0.transaction_amt',
                     buy_sell1='BUY_SELL_FLAG_POOL1',
                     buy=-1,
                     sell=1,
                     buy_sell2='BUY_SELL_FLAG_POOL2',
                     date='DATE',
                     time='time'):
    
    # Need to Decouple Pool 1 and Pool 2 into distinct Time Sorted Values
    temp_0 = df[[date,time,trader,amount,target,buy_sell1]].rename(columns={trader:'TRADER',amount:'VALUE',buy_sell1:'BUY_SELL'})
    temp_1 = df[[date,time,trader1,amount1,target,buy_sell2]].rename(columns={trader1:'TRADER',amount1:'VALUE',buy_sell2:'BUY_SELL'})
    tran_df = pd.concat([temp_0,temp_1])
    
    final_df = tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc=sum).reset_index().rename(columns={buy:'PURCHASE_VALUE',sell:"SELL_VALUE"})
    final_df = final_df.merge(tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc='count').reset_index().rename(columns={buy:'PURCHASE_VOLUME',sell:"SELL_VOLUME"}),on='TRADER',how='left')
    
    arb_tran_df = tran_df[tran_df[target]==1].copy()
    
    final_df = final_df.merge(arb_tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc=sum).reset_index().rename(columns={buy:'PURCHASE_VALUE_ARB',sell:"SELL_VALUE_ARB"}),on='TRADER',how='left')
    final_df = final_df.merge(arb_tran_df.pivot_table(index='TRADER',columns='BUY_SELL',values='VALUE',aggfunc='count').reset_index().rename(columns={buy:'PURCHASE_VOLUME_ARB',sell:"SELL_VOLUME_ARB"}),on='TRADER',how='left')
    
    
    final_df['ARB_PURCHASE_VAL'] = final_df['PURCHASE_VALUE_ARB']/final_df['PURCHASE_VALUE']
    final_df['ARB_PURCHASE_VOL'] = final_df['PURCHASE_VOLUME_ARB']/final_df['PURCHASE_VOLUME']
    final_df['ARB_SELL_VAL'] =     final_df['SELL_VALUE_ARB']/final_df['SELL_VALUE']
    final_df['ARB_SELL_VOL'] =     final_df['SELL_VOLUME_ARB']/final_df['SELL_VOLUME']
    
    final_df['TOTAL_VOLUME'] =  final_df['PURCHASE_VOLUME'] + final_df['SELL_VOLUME']
    final_df['TOTAL_VALUE']  =  final_df['PURCHASE_VALUE']  + final_df['SELL_VALUE']
    
    return tran_df,final_df


def RollingWindow(df,
                  new_col_name,
                  value_col,
                  net_volume_col,
                  abs_val_col,
                  trader_col,
                  time_col,
                  minutes=60,
                  type='trader',
                  calculation_type='val'):
    
    '''
    
    net_volume_column - For instances where direction of transaction matters, enables a user to calculate
    
    '''
    
    # Create Copy
    
    minutes_str = f"{minutes}min"
    new_col_vol = f"{new_col_name}_total_vol"
    new_col_val = f"{new_col_name}_net_val"
    new_col_net_vol = f"{new_col_name}_net_position"
    new_abs_val = f"{abs_val_col}_total_val"

    df=df.copy()    
    df['RECORD_COUNT']=1

    df1 = df[[value_col,net_volume_col,trader_col,time_col,abs_val_col,'RECORD_COUNT']]

    df1 = df1.groupby([trader_col,time_col]).sum().reset_index()
    
    df1 = df1.set_index(time_col).sort_index()

    if type=='summary':
        pass # Build out if needed.
            
    else:
        val = df1.groupby(trader_col)[value_col].rolling(window=minutes_str).sum().reset_index().rename(columns={value_col:new_col_val})
        vol = df1.groupby(trader_col)['RECORD_COUNT'].rolling(window=minutes_str).sum().reset_index().rename(columns={'RECORD_COUNT':new_col_vol})
        abs_val = df1.groupby(trader_col)[abs_val_col].rolling(window=minutes_str).sum().reset_index().rename(columns={abs_val_col:new_abs_val})
        net = df1.groupby(trader_col)[net_volume_col].rolling(window=minutes_str).sum().reset_index().rename(columns={net_volume_col:new_col_net_vol})
        return df.merge(val,on=[time_col,trader_col],how='left').merge(vol,on=[time_col,trader_col],how='left').merge(net,on=[time_col,trader_col],how='left').merge(abs_val,on=[time_col,trader_col],how='left').drop(new_abs_val,axis=1)




def brackets(df,column_name,new_column_name,desired_list=[0,10,20,30,40,50,60,70,80,90,100],
                      less_text='Less than',other_text='Between',greater_text='Greater than'):
    
    '''
    Purpose: Simple pre-defined formula to create STR definiton of Value Bucket

    Input: DataFrame, Column Name to Evaluate (No Format), New Column Name (No Format)
    
    Constraint: If Last Number is Lower bound constraint, then constraint = "lower", this duplicates the last item in the
    list and ensures that Valuation is completed correctly. Else it is ignored.

    List of values to evaluate, Maximum 10 values, can do less
    
    Default: list evenly spaced between 0 - 100

    Notes: Should consider enhancing a output list of easy filtering.

    '''
 
    desired_list.append(desired_list[-1:][0])
    
    condition = []
    value = []

    for count,i in enumerate(desired_list):
        if count == 0:
            condition.append(df[column_name]<=i)
        elif count == len(desired_list)-1:
            condition.append(df[column_name]>=i)
        else:
            condition.append(df[column_name]<=i)
        
    for count,i in enumerate(desired_list):
        if count == 0:
            value.append(f"{less_text} {i:,}")
        elif count == len(desired_list)-1:
            value.append(f"{greater_text} {i:,}")
        else:
            value.append(f"{other_text} {desired_list[count-1]:,} and {desired_list[count]:,}")
              
    df[new_column_name]=np.select(condition,value)
    
    return df


def Data_Dict_Missing_Cols(df,
                           data_dict,
                           extraction_type='column'):
    '''
    Function to compare a Data Dictionary, in the format of a Dataframe, to the columns, index, or values in a row of a dataframe.
    
    
    Parameters:
        df: DataFrame
        data_dict: A copy of Data Dictionary, file_reference: data_dictionary_v2.csv
        extraction_type: method of pulling information out of data dictionary, specifically it takes all values in a columnar method.


    Returns:
        Dataframe representing Columns which appear in the dataframe without a corresponding record in the data dictionary.

    '''
    if extraction_type == 'column':
        df_cols = pd.DataFrame(df.columns,columns=['FIELD_NAME'])

        final_df = df_cols.merge(data_dict,on='FIELD_NAME',how='left')
        print(f"Number of Missing Records:{len(final_df[final_df['FIELD_NAME'].isnull()])}")

        return final_df


def TransactionUniqueinDataset(df,
                               new_col_name,
                               tran_col_name):
    '''
    Purpose:
        Function to identify when a particular value appears consecutively in a dataframe, and where it does to return a binary 0. Utilized
        for the purposes of aggregating calculations in a columnar method, while removing duplciated records which appear in the dataframe.

        Data is expected to be in sorted order, as such the only record which is considered for evaluation is the previous record. Function 
        does correctly identify records which occur more than once concurrently.

    Parameters: 
        df: Dataframe
        new_col_name: Name of New Binary Flag Column which will be added to dataframe
        tran_col_name: Transaction Column Name, name of column which will be tested for uniquness.
    
    
    '''
    df[new_col_name] = df[tran_col_name] != df[tran_col_name].shift(1)
    df[new_col_name] = df[new_col_name].astype(int)


def CummulativeSum(df,
                   new_col_name,
                   cummulative_cols,
                   sum_col_name,
                   count_only=0,
                   binary_unique_tran_flag=''):
    '''
    

    '''

    if len(binary_unique_tran_flag)>0:
        df[new_col_name] = 0
        if count_only ==0:
            df['calculation'] = df[binary_unique_tran_flag]*df[sum_col_name]
        else:
            df['calculation'] = df[binary_unique_tran_flag]
        df[new_col_name] = df.groupby(cummulative_cols)['calculation'].cumsum()
        df.drop('calculation',axis=1,inplace=True)

    else:
        if count_only==1:
            df[new_col_name] = df[sum_col_name].cumsum()

        else:
            df[new_col_name] = df.groupby(cummulative_cols)[sum_col_name].cumsum()    
   

def CummulativeCount(df,
                     new_col_name,
                     cummulative_cols,
                     binary_unique_tran_flag=''):
    '''
    Can repalce with CummualtiveSum, simply count the Tran_flag. But it works so leave for now.
    
    '''

    if len(binary_unique_tran_flag)>0:
        df[new_col_name] = 0
        df['temp'] = df.groupby(cummulative_cols).cumcount()+1        
        # Reset Cummulative Value to the previous value if the transaction is NOT unique. Preventing Duplication
        df[new_col_name] = np.where(df[binary_unique_tran_flag]==1,df['temp'],df[new_col_name].shift())
        df[new_col_name] = df[new_col_name].replace(to_replace=0, value=pd.NA).ffill().fillna(0).astype(int)
    else:
        df[new_col_name] = 0
        df[new_col_name] = df.groupby(cummulative_cols).cumcount()+1   


def NEW_VARIABLE_CREATE(df):

    temp_df = df.copy()

    temp_df['TRADE_POOL1'] = np.where(temp_df['p1.t0_amount']<0,'Purchase','Sell')
    temp_df['TRADE_POOL2'] = np.where(temp_df['p0.t0_amount']<0,'Purchase','Sell')
    temp_df['TRADE_POOL_1_SELL_FLAG'] = np.where(temp_df['TRADE_POOL1']=='Sell',1,0)
    temp_df['TRADE_POOL_2_SELL_FLAG'] = np.where(temp_df['TRADE_POOL2']=='Sell',1,0)

    temp_df['TRADER_POOL1'] = np.where(temp_df['from_trade1'].notnull(),temp_df['from_trade1'],temp_df['p1.sender'])
    temp_df['TRADER_POOL2'] = np.where(temp_df['from_trade2'].notnull(),temp_df['from_trade2'],temp_df['p0.sender'])

    temp_df['SMART_CONTRACT_POOL1'] = np.where(temp_df['to_trade1'].notnull(),temp_df['to_trade1'],temp_df['p1.recipient'])
    temp_df['SMART_CONTRACT_POOL2'] = np.where(temp_df['to_trade2'].notnull(),temp_df['to_trade2'],temp_df['p0.recipient'])


    temp_df.sort_values('timestamp',inplace=True)

    TransactionUniqueinDataset(temp_df,
                               'Tran1Unique',
                               'p1.transaction_id')

    # Create Unique Transaction ID to avoid duplication for Pool 2
    TransactionUniqueinDataset(temp_df,
                               'Tran2Unique',
                               'p0.transaction_id')

    temp_df['p1.t0_amount_unique'] = temp_df['p1.t0_amount']*temp_df['Tran1Unique']
    temp_df['p0.t0_amount_unique'] = temp_df['p0.t0_amount']*temp_df['Tran2Unique']
    temp_df['p1.t0_amount_abs'] = temp_df['p1.t0_amount_abs']*temp_df['Tran1Unique']
    temp_df['p0.t0_amount_abs'] = temp_df['p0.t0_amount_abs']*temp_df['Tran2Unique']

    temp_df['BUY_SELL_BINARY1'] = np.where(temp_df['TRADE_POOL1']=='Sell',1,-1)*temp_df['Tran1Unique'].astype(np.int64)
    temp_df['BUY_SELL_BINARY2'] = np.where(temp_df['TRADE_POOL2']=='Sell',1,-1)*temp_df['Tran2Unique'].astype(np.int64)

    temp_df['BUY_SELL_BINARY1_unique'] = temp_df['BUY_SELL_BINARY1'] * temp_df['Tran1Unique']
    temp_df['BUY_SELL_BINARY2_unique'] = temp_df['BUY_SELL_BINARY2'] * temp_df['Tran2Unique']

    CummulativeSum(temp_df,
                   new_col_name='POOL_1_HISTORICAL_VOL',
                   cummulative_cols=[],
                   sum_col_name='Tran1Unique',
                   count_only=1,
                   binary_unique_tran_flag='')

    CummulativeSum(temp_df,
                   new_col_name='POOL_2_HISTORICAL_VOL',
                   cummulative_cols=[],
                   sum_col_name='Tran2Unique',
                   count_only=1,
                   binary_unique_tran_flag='')

    
    # Used slightly different function to due caluclation necessary.
    temp_df['POOL_1_HISTORICAL_TRADE_VAL'] = temp_df['p1.t0_amount_abs'] * temp_df['Tran1Unique']
    temp_df['POOL_1_HISTORICAL_TRADE_VAL'] = temp_df['POOL_1_HISTORICAL_TRADE_VAL'].cumsum()
    
    temp_df['POOL_2_HISTORICAL_TRADE_VAL'] = temp_df['p0.t0_amount_abs'] * temp_df['Tran2Unique']
    temp_df['POOL_2_HISTORICAL_TRADE_VAL'] = temp_df['POOL_2_HISTORICAL_TRADE_VAL'].cumsum()

    # POOL Level Daily Value and Volume
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_1_DAILY_VOL',
                   cummulative_cols=['DATE'],
                   sum_col_name='Tran1Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_2_DAILY_VOL',
                   cummulative_cols=['DATE'],
                   sum_col_name='Tran2Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran2Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_1_DAILY_VAL',
                   cummulative_cols=['DATE'],
                   sum_col_name='p1.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_2_DAILY_VAL',
                   cummulative_cols=['DATE'],
                   sum_col_name='p0.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')
   
    # Trader Level Daily Value and Volume


    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL1_DAILY_VOL',
                   cummulative_cols=['DATE','TRADER_POOL1'],
                   sum_col_name='Tran1Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL2_DAILY_VOL',
                   cummulative_cols=['DATE','TRADER_POOL2'],
                   sum_col_name='Tran2Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran2Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL1_DAILY_VAL',
                   cummulative_cols=['DATE','TRADER_POOL1'],
                   sum_col_name='p1.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL2_DAILY_VAL',
                   cummulative_cols=['DATE','TRADER_POOL2'],
                   sum_col_name='p0.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')



    ####
    # Added Jul 14.
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL1_HISTORICAL_VOL',
                   cummulative_cols=['TRADER_POOL1'],
                   sum_col_name='Tran1Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL2_HISTORICAL_VOL',
                   cummulative_cols=['TRADER_POOL2'],
                   sum_col_name='Tran2Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran2Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL1_HISTORICAL_VAL',
                   cummulative_cols=['TRADER_POOL1'],
                   sum_col_name='p1.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='TRADER_POOL2_HISTORICAL_VAL',
                   cummulative_cols=['TRADER_POOL2'],
                   sum_col_name='p0.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')

    #####
    # Added Jul 12

    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_1_HISTORICAL_VOL',
                   cummulative_cols=['SMART_CONTRACT_POOL1'],
                   sum_col_name='Tran1Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_2_HISTORICAL_VOL',
                   cummulative_cols=['SMART_CONTRACT_POOL2'],
                   sum_col_name='Tran2Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran2Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_1_HISTORICAL_VAL',
                   cummulative_cols=['SMART_CONTRACT_POOL1'],
                   sum_col_name='p1.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_2_HISTORICAL_VAL',
                   cummulative_cols=['SMART_CONTRACT_POOL2'],
                   sum_col_name='p0.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')

    #####


    
    # Smart Contract Level Daily Value and Volume

    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_1_DAILY_VOL',
                   cummulative_cols=['DATE','SMART_CONTRACT_POOL1'],
                   sum_col_name='Tran1Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_2_DAILY_VOL',
                   cummulative_cols=['DATE','SMART_CONTRACT_POOL2'],
                   sum_col_name='Tran2Unique',
                   count_only=1,
                   binary_unique_tran_flag='Tran2Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_1_DAILY_VAL',
                   cummulative_cols=['DATE','SMART_CONTRACT_POOL1'],
                   sum_col_name='p1.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='SMART_CONTRACT_POOL_2_DAILY_VAL',
                   cummulative_cols=['DATE','SMART_CONTRACT_POOL2'],
                   sum_col_name='p0.t0_amount_abs',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')

    
    # Net Position Vol and Val. 

    temp_df['POOL_1_HISTORICAL_TRANSACTION_POSITION'] = temp_df['BUY_SELL_BINARY1'].cumsum().astype(np.int64)
    temp_df['POOL_2_HISTORICAL_TRANSACTION_POSITION'] = temp_df['BUY_SELL_BINARY2'].cumsum().astype(np.int64)

    temp_df['POOL_1_HISTORICAL_POSITION_VALUE'] = temp_df['p1.t0_amount_unique'].cumsum().astype(np.int64)
    temp_df['POOL_2_HISTORICAL_POSITION_VALUE'] = temp_df['p0.t0_amount_unique'].cumsum().astype(np.int64)    

    # Daily Position Vol and Val

    CummulativeSum(temp_df,
                   new_col_name='POOL_1_DAILY_VOL_POSITION',
                   cummulative_cols=['DATE'],
                   sum_col_name='BUY_SELL_BINARY1',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_2_DAILY_VOL_POSITION',
                   cummulative_cols=['DATE'],
                   sum_col_name='BUY_SELL_BINARY2',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_1_DAILY_VAL_POSITION',
                   cummulative_cols=['DATE'],
                   sum_col_name='p1.t0_amount',
                   count_only=0,
                   binary_unique_tran_flag='Tran1Unique')
    
    CummulativeSum(temp_df,
                   new_col_name='POOL_2_DAILY_VAL_POSITION',
                   cummulative_cols=['DATE'],
                   sum_col_name='p0.t0_amount',
                   count_only=0,
                   binary_unique_tran_flag='Tran2Unique')
    
    temp_df['target'] = np.where(temp_df['swap_go_nogo']==True,1,0)
        
    AggregatePoolCalculations(temp_df)
    
    temp_df = RollingWindow(temp_df,
                      new_col_name='TRADER1_60M',
                      value_col='p1.t0_amount_unique',
                      net_volume_col='BUY_SELL_BINARY1',
                      abs_val_col='p1.t0_amount_abs',
                      trader_col='TRADER_POOL1',
                      time_col='time')
    
    temp_df = RollingWindow(temp_df,
                      new_col_name='TRADER2_60M',
                      value_col='p0.t0_amount_unique',
                      net_volume_col='BUY_SELL_BINARY2',
                      abs_val_col='p0.t0_amount_abs',
                      trader_col='TRADER_POOL2',
                      time_col='time')
    
    temp_df = RollingWindow(temp_df,
                      new_col_name='SMART_CONTRACT_POOL1_60M',
                      value_col='p1.t0_amount_unique',
                      net_volume_col='BUY_SELL_BINARY1',
                      abs_val_col='p1.t0_amount_abs',
                      trader_col='SMART_CONTRACT_POOL1',
                      time_col='time')
    
    temp_df = RollingWindow(temp_df,
                      new_col_name='SMART_CONTRACT_POOL2_60M',
                      value_col='p0.t0_amount_unique',
                      net_volume_col='BUY_SELL_BINARY1',
                      abs_val_col='p0.t0_amount_abs',
                      trader_col='SMART_CONTRACT_POOL2',
                      time_col='time')
    return temp_df


def TraderDefinition(df,
                     new_column_name= 'TRADER_CLASSIFICATION',
                     trader_total_trade_value='TOTAL_VALUE',
                     trader_net_trade_position='NET_POSITION'):
    '''
    Function which creates unique definitions at a trader level, based on the Total Historical ACtivity of Trader in Individual Activity
    Pools.

    Parameters:
        DataFrame (Note this Dataframe should be at a Consolidiated Trader Level and include all historical information).
        trader_total_trade_value: Aggregate Trade Value of Trader in Consolidated Pool Activity
        trader_net_trade_position: Aggregate Trade Position of Trader in Consolidated Pool Activity
        
        
    
    
    '''

    condition = [df[trader_total_trade_value]<=20000,
                 (df[trader_total_trade_value]>100000) &((df[trader_net_trade_position]<10000)&(df[trader_net_trade_position]>-10000)),
                 df[trader_total_trade_value]<=100000,
                 df[trader_total_trade_value]>100000]

    value = ['Retail Trader',
             'Trader',
             'Retail Investor',
             'Investor']

    df[new_column_name] = np.select(condition,value,'Not Defined')  


def ConvertToETHPrice(df,
                      new_column_name,
                      column,
                      current_value='Gwei'):
    
    if current_value=='Wei':
        df[new_column_name] = df[column].apply(lambda x:x/10**18)
        df[new_column_name] = df[column]
    elif current_value == 'Gwei':
        df[new_column_name] = df[column].apply(lambda x:x/10**9)
    elif current_value=='sq_96':
        df[new_column_name] = df[column].apply(lambda x:x/2**96)




def GenerateUniqueTradeDataset(df,
                               trader='TRADER_POOL1',
                               trader1='TRADER_POOL2',
                               amount='p1.t0_amount_unique',
                               amount1='p0.t0_amount_unique',
                               vol_unique1 = 'BUY_SELL_BINARY1_unique',
                               vol_unique2 = 'BUY_SELL_BINARY2_unique',
                               gasUsed=['p1.gasUsed','p0.gasUsed'],
                               gasPrice=['p1.gasPrice_ETH','p0.gasPrice_ETH'],
                               target='target',
                               date='DATE',
                               time='time'):

    '''
    Function which takes finalized Dataset, then splits into individual pools and deduplicates trades, such that all Unique Trades level
    review of all trades in pool 1 and pool 2 is available.

    Parameters
        df: DataFrame
        trader: Column Identifies Trader from Pool 1
        trader1: Column Identifying Trader from Pool 2
        amount: USDC Equivalent of Trade from Pool 1
        amount1; USDC Equivalent of Trade from Pool 2
        vol_unique1: Binary flag from pool 1 identifying whether trade from pool 1 is unique and should be included.
        vol_unique2: Binary flag from pool 2 identifying whether trade from pool 1 is unique and should be included.
        date: Date of Transaction
        time: Time of Transaction

    Returns:
        Dataframe with combined values into a single Dataframe, which considers all transactions. 
        
    '''
    
    
    # Need to Decouple Pool 1 and Pool 2 into distinct Time Sorted Values
    temp_0 = df[[date,time,trader,amount,vol_unique1,gasUsed[0],gasPrice[0],target]].rename(columns={trader:'TRADER',amount:'VALUE',vol_unique1:'UNI_COUNT',gasPrice[0]:'GAS_PRICE',gasUsed[0]:'GAS_USED'})
    temp_1 = df[[date,time,trader1,amount1,vol_unique2,gasUsed[1],gasPrice[1],target]].rename(columns={trader1:'TRADER',amount1:'VALUE',vol_unique2:'UNI_COUNT',gasPrice[1]:'GAS_PRICE',gasUsed[1]:'GAS_USED'})
    temp_ = pd.concat([temp_0,temp_1])
    return temp_[temp_['UNI_COUNT']!=0]


def format_number(n):
    if n <-1_000_000_000:
        return f"{int(n / 1_000_000_000)}B" if n % 1_000_000_000 == 0 else f"{n / 1_000_000_000:.1f}B"
    elif n <= -1_000_000:
        return f"{int(n / 1_000_000)}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    elif n <= -1_000:
        return f"{int(n / 1_000)}K" if n % 1_000 == 0 else f"{n / 1_000:.1f}K"

    elif n >= 1_000_000_000:
        return f"{int(n / 1_000_000_000)}B" if n % 1_000_000_000 == 0 else f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{int(n / 1_000_000)}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{int(n / 1_000)}K" if n % 1_000 == 0 else f"{n / 1_000:.1f}K"
    else:
        return str(n)

def brackets2(df, column_name, new_column_name, desired_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
             less_text='Less than', greater_text='Greater than'):
    '''
    Purpose: Simple pre-defined formula to create STR definition of Value Bucket

    Input: DataFrame, Column Name to Evaluate (No Format), New Column Name (No Format)
    
    Constraint: If Last Number is Lower bound constraint, then constraint = "lower", this duplicates the last item in the
    list and ensures that Valuation is completed correctly. Else it is ignored.

    List of values to evaluate, Maximum 10 values, can do less
    
    Default: list evenly spaced between 0 - 100

    Notes: Should consider enhancing a output list of easy filtering.
    '''
    
    # Add a duplicate of the last item in the desired_list
    desired_list.append(desired_list[-1:][0])
    
    condition = []
    value = []

    for count, i in enumerate(desired_list):
        if count == 0:
            condition.append(df[column_name] <= i)
        elif count == len(desired_list) - 1:
            condition.append(df[column_name] >= i)
        else:
            condition.append(df[column_name] <= i)
        
    for count, i in enumerate(desired_list):
        if count == 0:
            value.append(f"{less_text} {format_number(i)}")
        elif count == len(desired_list) - 1:
            value.append(f"{greater_text} {format_number(i)}")
        else:
            value.append(f"{format_number(desired_list[count-1])}-{format_number(desired_list[count])}")
              
    df[new_column_name] = np.select(condition, value)
    
    return df


def LAGGED_VALUE(df,
                 time_column,
                 lagged_column,
                 lag_in_minutes=5):

    new_df = df[[lagged_column,time_column]].copy()

    new_df = new_df.sort_values(time_column).drop_duplicates(time_column)
    

    # Replace TIme Column with time in 5 minutes
    new_df[time_column] = new_df[time_column] + pd.Timedelta(minutes=lag_in_minutes)

    final_df = pd.merge_asof(df.sort_values(time_column), new_df, on=time_column, direction='backward', suffixes=('', f'_{lag_in_minutes}min_lag'))
    final_df[f'{lagged_column}_{lag_in_minutes}min_lag'] = final_df[f'{lagged_column}_{lag_in_minutes}min_lag'].fillna(0)

    return final_df




