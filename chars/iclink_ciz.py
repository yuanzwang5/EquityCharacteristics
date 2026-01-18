"""
*********************************************************************************************
************************** PYTHON VERSION OF WRDS RESEARCH MACROS **************************
*********************************************************************************************
Original WRDS Macro: ICLINK_CIZ
Summary: Create IBES-CRSP Link Table
Original Author: Rabih Moussawi, WRDS
Original Date: September 25, 2006
SAS Update: November 2024 by Freda Drechsler for CRSP CIZ data format
Python Update: July 2025 by Yuning598

Variables:
    - IBES ID and CRSP ID are IBES and CRSP Names Datasets
    - Outset: IBES-CRSP link table output dataset
*********************************************************************************************
"""

import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import *
import pyarrow.feather as feather
import os
import time
import re
from fuzzywuzzy import fuzz

def improved_spedis(str1, str2):
    """
    More accurate implementation of SAS's SPEDIS function
    
    SPEDIS in SAS returns values between 0 (identical) and ~100+ (very different)
    This function combines edit distance with length penalties similar to SPEDIS
    
    Args:
        str1, str2: Input strings to compare
    
    Returns:
        int: SPEDIS-like distance score (0 is perfect match)
    """
    if str1 is None or str2 is None:
        return 100
        
    # Convert to strings and lowercase
    s1 = str(str1).lower().strip()
    s2 = str(str2).lower().strip()
    
    # Apply SAS-like preprocessing
    # Remove punctuation to better match SAS SPEDIS behavior
    s1 = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', ' ', s1)
    s2 = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', ' ', s2)
    
    # Remove extra spaces
    s1 = re.sub(r'\s+', ' ', s1).strip()
    s2 = re.sub(r'\s+', ' ', s2).strip()
    
    # If either string is empty after processing
    if not s1 or not s2:
        return 100
    
    # Compute string similarity using token_set_ratio which is robust to word order
    # But convert back to a distance metric (0-100 scale)
    similarity = fuzz.token_set_ratio(s1, s2)
    distance = 100 - similarity
    
    # Add penalties for length differences like SAS SPEDIS does
    len_diff = abs(len(s1) - len(s2))
    length_penalty = min(20, len_diff * 3)  # Cap the penalty
    
    # Final score
    spedis_value = distance + length_penalty
    return min(100, spedis_value)  # Cap at 100

def iclink_ciz(ibes_id='ibes.id', crsp_id='crsp.stocknames_v2', outset='iclink.csv', 
               conn=None, close_conn=True, save_feather=True, debug=False):
    """
    Create IBES-CRSP Link Table
    
    Args:
        ibes_id (str): IBES ID dataset name in WRDS
        crsp_id (str): CRSP ID dataset name in WRDS
        outset (str): Output file path for CSV
        conn (wrds.Connection, optional): WRDS connection object. If None, creates a new connection
        close_conn (bool): Whether to close the WRDS connection when done
        save_feather (bool): Whether to save the result as feather file as well
        debug (bool): Print debug information
        
    Returns:
        DataFrame: IBES-CRSP link table
    """
    print("\n### START. Creating IBES-CRSP Link Table: ICLINK")
    print(f"## IBES NAMES (ID) Dataset Used: {ibes_id}")
    print(f"## CRSP NAMES (ID) Dataset Used: {crsp_id}")
    
    # Connect to WRDS if connection not provided
    if conn is None:
        print("Connecting to WRDS...")
        conn = wrds.Connection()
        
    print("## Step1: Linking using CUSIPs...")
    
    #########################
    # Step 1: Link by CUSIP #
    #########################
    
    # 1.1 IBES: Get the list of IBES Tickers for US firms in IBES
    _ibes1 = conn.raw_sql(f"""
                      select ticker, cusip, cname, sdates from {ibes_id}
                      where usfirm=1 and cusip != ''
                      """)
    
    if debug:
        print(f"IBES data columns: {_ibes1.columns.tolist()}")
    
    # Ensure column names are lowercase
    _ibes1.columns = [col.lower() for col in _ibes1.columns]
    
    # Create first and last 'start dates' for a given cusip
    _ibes1_date = _ibes1.groupby(['ticker', 'cusip']).sdates.agg(['min', 'max'])\
        .reset_index().rename(columns={'min': 'fdate', 'max': 'ldate'})
    
    # Merge fdate ldate back to _ibes1 data
    _ibes2 = pd.merge(_ibes1, _ibes1_date, how='left', on=['ticker', 'cusip'])
    _ibes2 = _ibes2.sort_values(by=['ticker', 'cusip', 'sdates'])
    
    # Keep only the most recent company name
    # Determined by having sdates = ldate
    _ibes2 = _ibes2.loc[_ibes2.sdates == _ibes2.ldate].drop(['sdates'], axis=1)
    
    # 1.2 CRSP: Get all permno-ncusip combinations
    _crsp1 = conn.raw_sql(f"""
                      select permno, cusip, issuernm, namedt, nameenddt
                      from {crsp_id}
                      where cusip != ''
                      """)
    
    # Ensure column names are lowercase
    _crsp1.columns = [col.lower() for col in _crsp1.columns]
    
    # First namedt
    _crsp1_fnamedt = _crsp1.groupby(['permno', 'cusip']).namedt.min().reset_index()
    
    # Last nameenddt
    _crsp1_lnameenddt = _crsp1.groupby(['permno', 'cusip']).nameenddt.max().reset_index()
    
    # Merge both
    _crsp2 = pd.merge(_crsp1_fnamedt, _crsp1_lnameenddt, on=['permno', 'cusip'], how='inner')
    
    # Get most recent company name for each permno-cusip combination
    _crsp1_latest = _crsp1.sort_values('nameenddt', ascending=False)\
                    .groupby(['permno', 'cusip']).first().reset_index()
    
    # Add company name to _crsp2
    _crsp2 = pd.merge(_crsp2, _crsp1_latest[['permno', 'cusip', 'issuernm']], 
                     on=['permno', 'cusip'], how='left')
    
    # 1.3 Create CUSIP Link Table
    # Link by full cusip, company names and dates
    _link1_1 = pd.merge(_ibes2, _crsp2, how='inner', left_on='cusip', right_on='cusip')\
        .sort_values(['ticker', 'permno', 'ldate'])
    
    # Keep link with most recent company name for each TICKER-PERMNO pair
    # Important: SAS uses BY TICKER PERMNO; if last.permno to get most recent company name
    _link1_2 = _link1_1.sort_values(['ticker', 'permno', 'ldate'], ascending=[True, True, False])\
                       .groupby(['ticker', 'permno']).first().reset_index()
    
    # Calculate name distance using our improved_spedis function (similar to SAS SPEDIS)
    _link1_2['name_dist'] = _link1_2.apply(lambda x: 
                                           min(improved_spedis(x['cname'], x['issuernm']),
                                               improved_spedis(x['issuernm'], x['cname'])), 
                                          axis=1)
    
    # Score using same logic as SAS
    def score_cusip_link(row):
        # Exactly match SAS logic:
        # if (not ((ldate < namedt) or (fdate > nameenddt))) and name_dist < 30 then SCORE = 0;
        # else if (not ((ldate < namedt) or (fdate > nameenddt))) then score = 1;
        # else if name_dist < 30 then SCORE = 2;
        # else SCORE = 3;
        
        date_match = not ((row['ldate'] < row['namedt']) or (row['fdate'] > row['nameenddt']))
        name_match = row['name_dist'] < 30
        
        if date_match and name_match:
            return 0
        elif date_match:
            return 1
        elif name_match:
            return 2
        else:
            return 3
    
    # Assign scores
    _link1_2['score'] = _link1_2.apply(score_cusip_link, axis=1)
    
    # Keep only necessary columns
    _link1_2 = _link1_2[['ticker', 'permno', 'cname', 'issuernm', 'name_dist', 'score']]
    
    print("## Step2: Linking using TICKERs...")
    
    ##########################
    # Step 2: Link by TICKER #
    ##########################
    
    # Find links for the remaining unmatched cases using Exchange Ticker
    # Identify remaining unmatched cases - match SAS logic exactly
    _nomatch1 = pd.DataFrame({'ticker': _ibes1['ticker'].unique()})
    _nomatch1 = _nomatch1[~_nomatch1['ticker'].isin(_link1_2['ticker'])]
    
    # Add IBES identifying information
    ibesid = conn.raw_sql(f"""select ticker, cname, oftic, sdates, cusip from {ibes_id}
                            where oftic is not null and oftic != ''""")
    
    # Ensure column names are lowercase
    ibesid.columns = [col.lower() for col in ibesid.columns]
    
    _nomatch2 = pd.merge(_nomatch1, ibesid, how='inner', on=['ticker'])
    
    # Create first and last 'start dates' for Exchange Tickers
    _nomatch3_dates = _nomatch2.groupby(['ticker', 'oftic']).sdates.agg(['min', 'max'])\
        .reset_index().rename(columns={'min': 'fdate', 'max': 'ldate'})
    
    _nomatch3 = pd.merge(_nomatch2, _nomatch3_dates, how='left', on=['ticker', 'oftic'])
    
    # Keep only the most recent company name
    # Using exact SAS logic: by ticker oftic; if last.oftic;
    _nomatch3 = _nomatch3.sort_values(['ticker', 'oftic', 'sdates'], ascending=[True, True, True])
    _nomatch3 = _nomatch3.groupby(['ticker', 'oftic']).last().reset_index()
    
    # Get entire list of CRSP stocks with Exchange Ticker information
    _crsp_n1 = conn.raw_sql(f"""select ticker, issuernm, permno, cusip, namedt, nameenddt
                             from {crsp_id}
                             where ticker is not null and ticker != ''""")
    
    # Ensure column names are lowercase
    _crsp_n1.columns = [col.lower() for col in _crsp_n1.columns]
    _crsp_n1.rename(columns={'cusip': 'crsp_cusip'}, inplace=True)
    _crsp_n1 = _crsp_n1.sort_values(by=['permno', 'ticker', 'namedt'])
    
    # Arrange effective dates for link by Exchange Ticker
    _crsp_n2_dates = _crsp_n1.groupby(['permno', 'ticker']).agg({
        'namedt': 'min',
        'nameenddt': 'max'
    }).reset_index()
    
    # Get most recent company name for each permno-ticker combination
    _crsp_n1_latest = _crsp_n1.sort_values(['permno', 'ticker', 'nameenddt'], ascending=[True, True, False])\
                    .groupby(['permno', 'ticker']).first().reset_index()
    
    # Create the final CRSP data with date ranges
    _crsp_n2 = pd.merge(_crsp_n2_dates, 
                       _crsp_n1_latest[['permno', 'ticker', 'issuernm', 'crsp_cusip']],
                       on=['permno', 'ticker'], how='left')
    
    # Rename ticker to avoid confusion with IBES TICKER
    _crsp_n2.rename(columns={'ticker': 'crsp_ticker'}, inplace=True)
    
    # Merge remaining unmatched cases using Exchange Ticker
    # Note: Use ticker date ranges as exchange tickers are reused overtime
    _link2_1 = pd.merge(_nomatch3, _crsp_n2, how='inner', left_on=['oftic'], right_on=['crsp_ticker'])
    _link2_1 = _link2_1[(_link2_1.ldate >= _link2_1.namedt) & (_link2_1.fdate <= _link2_1.nameenddt)]
    
    # Calculate name distance using our improved_spedis function
    _link2_1['name_dist'] = _link2_1.apply(lambda x: 
                                           min(improved_spedis(x['cname'], x['issuernm']),
                                               improved_spedis(x['issuernm'], x['cname'])), 
                                          axis=1)
    
    # Extract 6-digit CUSIPs for comparison
    _link2_2 = _link2_1.copy()
    _link2_2['cusip6'] = _link2_2['cusip'].str[:6]
    _link2_2['crsp_cusip6'] = _link2_2['crsp_cusip'].str[:6]
    
    # Score using company name, 6-digit CUSIP, and company name distance
    # Using exact SAS logic:
    # if substr(cusip,1,6)=substr(crsp_cusip,1,6) and name_dist < 30 then SCORE=0;
    # else if substr(cusip,1,6)=substr(crsp_cusip,1,6) then score = 4;
    # else if name_dist < 30 then SCORE = 5;
    # else SCORE = 6;
    def score_ticker_link(row):
        cusip6_match = row['cusip6'] == row['crsp_cusip6'] 
        name_match = row['name_dist'] < 30
        
        if cusip6_match and name_match:
            return 0
        elif cusip6_match:
            return 4
        elif name_match:
            return 5
        else:
            return 6
    
    # Assign scores
    _link2_2['score'] = _link2_2.apply(score_ticker_link, axis=1)
    
    # Sort by ticker and score (ascending)
    _link2_2 = _link2_2.sort_values(by=['ticker', 'score'])
    
    # Keep only the first record for each ticker (lowest score)
    # Using exact SAS logic: by ticker score; if first.ticker;
    _link2_3 = _link2_2.groupby('ticker').first().reset_index()
    
    # Keep only necessary columns (match SAS)
    _link2_3 = _link2_3[['ticker', 'permno', 'cname', 'issuernm', 'score']]
    
    print("## Step3: Finalizing Links and Scores...")
    
    #####################################
    # Step 3: Finalize Links and Scores #
    #####################################
    # Combine the output from both linking procedures
    
    iclink = pd.concat([_link1_2[['ticker', 'permno', 'cname', 'issuernm', 'score']], 
                        _link2_3], ignore_index=True)
    
    iclink = iclink.sort_values(by=['ticker', 'score', 'permno'])
    
    print(f"## Step4: Link Table {outset} Ready...")
    
    # Save output
    if outset.endswith('.csv'):
        iclink.to_csv(outset, index=False)
        print(f"Saved CSV output to {outset}")
    
    # Save feather file for faster loading in other Python programs
    if save_feather:
        feather_path = outset.replace('.csv', '.feather')
        feather.write_feather(iclink, feather_path)
        print(f"Saved feather output to {feather_path}")
    
    # Close connection if requested
    if close_conn and conn is not None:
        conn.close()
        print("WRDS connection closed")
    
    print("### DONE.")
    return iclink

if __name__ == "__main__":
    try:
        conn = wrds.Connection(wrds_username='phd22jm', wrds_password='jmwarwickap1998!')
        iclink = iclink_ciz(conn=conn, outset='iclink_ciz.csv', debug=False)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()