# Since some firms only have annual recording before 80s, we need to use annual data as merging benchmark in case
# there are some recordings are missing.

# After expanding the data from 1926, we need to make sure every stock at least have corresponding return.

import pandas as pd
import pickle as pkl
import pyarrow.feather as feather
from pandas.tseries.offsets import *
import numpy as np
import wrds

######################################################################
# read return data and fill the missing value in accounting files
conn = wrds.Connection()
print(f"Connected to WRDS successfully!")
crsp = conn.raw_sql("""
                    select a.mthprc, a.mthret, a.mthretx, a.shrout, a.mthvol, a.mthcaldt, a.permno, a.permco,
                    a.primaryexch, a.conditionaltype, a.TradingStatusFlg, a.mthretflg, a.mthprcflg,
                    a.issuernm, a.issuertype, a.securitytype, a.securitysubtype, a.sharetype, a.usincflg
                    from crsp.msf_v2 as a
                    where a.mthcaldt >= '01/01/1959'
                    """, date_cols=['mthcaldt'])

# equivalent to legacy code exchcd = 1, 2 or 3
crsp = crsp.loc[(crsp.primaryexch.isin(['N', 'A', 'Q'])) & \
                   (crsp.conditionaltype =='RW') & \
                   (crsp.tradingstatusflg =='A')]
# equivalent to legacy code shrcd = 10 or 11
crsp = crsp.loc[(crsp.sharetype=='NS') & \
                    (crsp.securitytype=='EQTY') & \
                    (crsp.securitysubtype=='COM') & \
                    (crsp.usincflg=='Y') & \
                    (crsp.issuertype.isin(['ACOR', 'CORP']))]
crsp.drop(['primaryexch', 'conditionaltype', 'tradingstatusflg', 'securitytype', 'securitysubtype',
           'sharetype', 'usincflg', 'issuertype'], axis=1, inplace=True)

crsp.rename(columns={'mthprc': 'prc', 'mthret': 'ret', 'mthretx': 'retx', 'mthcaldt': 'date'}, inplace=True)

crsp = crsp.dropna(subset=['ret', 'retx', 'prc'])

# change variable format to int
crsp[['permco', 'permno']] = crsp[['permco', 'permno']].astype(int)

# Line up date to be end of month
crsp['date'] = pd.to_datetime(crsp['date'])
crsp['jdate'] = crsp['date'] + MonthEnd(0)  # set all the date to the standard end date of month

crsp = crsp.dropna(subset=['prc'])
crsp['me'] = crsp['prc'].abs() * crsp['shrout']  # calculate market equity

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['jdate', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['jdate', 'permco', 'me'])
# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['jdate', 'permco'])
# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by=['permno', 'jdate']).drop_duplicates()

################## Added on 2025.02.23 ##################
crsp2['me'] = crsp2['me']/1000 # Change from thousand to million

crsp = crsp2.copy()
crsp = crsp.sort_values(by=['permno', 'date'])

crsp['ret'] = crsp['ret'].fillna(0)

crsp = crsp[['permno', 'jdate', 'ret', 'retx', 'me']]
crsp.columns = ['permno', 'jdate', 'ret_fill', 'retx_fill', 'me_fill']
######################################################################

with open('chars_a_accounting.feather', 'rb') as f:
    chars_a = feather.read_feather(f)
f.close()

chars_a = chars_a.dropna(subset=['permno'])
chars_a[['permno', 'gvkey']] = chars_a[['permno', 'gvkey']].astype(int)
chars_a['jdate'] = pd.to_datetime(chars_a['jdate'])
chars_a = chars_a.drop_duplicates(['permno', 'jdate'])

with open('beta.feather', 'rb') as f:
    beta = feather.read_feather(f)
f.close()

beta['permno'] = beta['permno'].astype(int)
beta['jdate'] = pd.to_datetime(beta['date']) + MonthEnd(0)
beta = beta[['permno', 'jdate', 'beta']]
beta = beta.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, beta, how='left', on=['permno', 'jdate'])

with open('rvar_capm.feather', 'rb') as f:
    rvar_capm = feather.read_feather(f)
f.close()

rvar_capm['permno'] = rvar_capm['permno'].astype(int)
rvar_capm['jdate'] = pd.to_datetime(rvar_capm['date']) + MonthEnd(0)
rvar_capm = rvar_capm[['permno', 'jdate', 'rvar_capm']]
rvar_capm = rvar_capm.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, rvar_capm, how='left', on=['permno', 'jdate'])

with open('rvar_mean.feather', 'rb') as f:
    rvar_mean = feather.read_feather(f)
f.close()

rvar_mean['permno'] = rvar_mean['permno'].astype(int)
rvar_mean['jdate'] = pd.to_datetime(rvar_mean['date']) + MonthEnd(0)
rvar_mean = rvar_mean[['permno', 'jdate', 'rvar_mean']]
rvar_mean = rvar_mean.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, rvar_mean, how='left', on=['permno', 'jdate'])

with open('rvar_ff3.feather', 'rb') as f:
    rvar_ff3 = feather.read_feather(f)
f.close()

rvar_ff3['permno'] = rvar_ff3['permno'].astype(int)
rvar_ff3['jdate'] = pd.to_datetime(rvar_ff3['date']) + MonthEnd(0)
rvar_ff3 = rvar_ff3[['permno', 'jdate', 'rvar_ff3']]
rvar_ff3 = rvar_ff3.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, rvar_ff3, how='left', on=['permno', 'jdate'])

with open('sue.feather', 'rb') as f:
    sue = feather.read_feather(f)
f.close()

sue['permno'] = sue['permno'].astype(int)
sue['jdate'] = pd.to_datetime(sue['date']) + MonthEnd(0)
sue = sue[['permno', 'jdate', 'sue']]
sue = sue.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, sue, how='left', on=['permno', 'jdate'])

with open('myre.feather', 'rb') as f:
    re = feather.read_feather(f)
f.close()

re['permno'] = re['permno'].astype(int)
re['jdate'] = pd.to_datetime(re['date']) + MonthEnd(0)
re = re[['permno', 'jdate', 're']]
re = re.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, re, how='left', on=['permno', 'jdate'])

with open('abr.feather', 'rb') as f:
    abr = feather.read_feather(f)
f.close()

abr['permno'] = abr['permno'].astype(int)
abr['jdate'] = pd.to_datetime(abr['date']) + MonthEnd(0)
abr = abr[['permno', 'jdate', 'abr']]
abr = abr.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, abr, how='left', on=['permno', 'jdate'])

with open('baspread.feather', 'rb') as f:
    baspread = feather.read_feather(f)
f.close()

baspread['permno'] = baspread['permno'].astype(int)
baspread['jdate'] = pd.to_datetime(baspread['date']) + MonthEnd(0)
baspread = baspread[['permno', 'jdate', 'baspread']]
baspread = baspread.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, baspread, how='left', on=['permno', 'jdate'])

with open('maxret.feather', 'rb') as f:
    maxret = feather.read_feather(f)
f.close()

maxret['permno'] = maxret['permno'].astype(int)
maxret['jdate'] = pd.to_datetime(maxret['date']) + MonthEnd(0)
maxret = maxret[['permno', 'jdate', 'maxret']]
maxret = maxret.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, maxret, how='left', on=['permno', 'jdate'])

with open('std_dolvol.feather', 'rb') as f:
    std_dolvol = feather.read_feather(f)
f.close()

std_dolvol['permno'] = std_dolvol['permno'].astype(int)
std_dolvol['jdate'] = pd.to_datetime(std_dolvol['date']) + MonthEnd(0)
std_dolvol = std_dolvol[['permno', 'jdate', 'std_dolvol']]
std_dolvol = std_dolvol.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, std_dolvol, how='left', on=['permno', 'jdate'])

with open('ill.feather', 'rb') as f:
    ill = feather.read_feather(f)
f.close()

ill['permno'] = ill['permno'].astype(int)
ill['jdate'] = pd.to_datetime(ill['date']) + MonthEnd(0)
ill = ill[['permno', 'jdate', 'ill']]
ill = ill.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, ill, how='left', on=['permno', 'jdate'])

with open('std_turn.feather', 'rb') as f:
    std_turn = feather.read_feather(f)
f.close()

std_turn['permno'] = std_turn['permno'].astype(int)
std_turn['jdate'] = pd.to_datetime(std_turn['date']) + MonthEnd(0)
std_turn = std_turn[['permno', 'jdate', 'std_turn']]
std_turn = std_turn.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, std_turn, how='left', on=['permno', 'jdate'])

with open('zerotrade.feather', 'rb') as f:
    zerotrade = feather.read_feather(f)
f.close()

zerotrade['permno'] = zerotrade['permno'].astype(int)
zerotrade['jdate'] = pd.to_datetime(zerotrade['date']) + MonthEnd(0)
zerotrade = zerotrade[['permno', 'jdate', 'zerotrade']]
zerotrade = zerotrade.drop_duplicates(['permno', 'jdate'])

chars_a = pd.merge(chars_a, zerotrade, how='left', on=['permno', 'jdate'])

# fill the return
chars_a = pd.merge(chars_a, crsp, how='left', on=['permno', 'jdate'])
chars_a['ret'] = np.where(chars_a['ret'].isnull(), chars_a['ret_fill'], chars_a['ret'])
chars_a['retx'] = np.where(chars_a['retx'].isnull(), chars_a['retx_fill'], chars_a['retx'])
chars_a['me'] = np.where(chars_a['me'].isnull(), chars_a['me_fill'], chars_a['me'])
# chars_a['exchcd'] = np.where(chars_a['exchcd'].isnull(), chars_a['exchcd_fill'], chars_a['exchcd'])
# chars_a['shrcd'] = np.where(chars_a['shrcd'].isnull(), chars_a['shrcd_fill'], chars_a['shrcd'])

chars_a = chars_a.dropna(subset=['permno', 'jdate', 'ret', 'retx'])
# chars_a = chars_a[((chars_a['exchcd'] == 1) | (chars_a['exchcd'] == 2) | (chars_a['exchcd'] == 3)) &
#                    ((chars_a['shrcd'] == 10) | (chars_a['shrcd'] == 11))]
chars_a = chars_a.loc[(chars_a.primaryexch.isin(['N', 'A', 'Q'])) & \
                   (chars_a.conditionaltype =='RW') & \
                   (chars_a.tradingstatusflg =='A')]
# equivalent to legacy code shrcd = 10 or 11
chars_a = chars_a.loc[(chars_a.sharetype=='NS') & \
                    (chars_a.securitytype=='EQTY') & \
                    (chars_a.securitysubtype=='COM') & \
                    (chars_a.usincflg=='Y') & \
                    (chars_a.issuertype.isin(['ACOR', 'CORP']))]

# save data
with open('chars_a_raw.feather', 'wb') as f:
    feather.write_feather(chars_a, f)
f.close()

########################################################################################################################
#     In order to keep the naming tidy, we need to make another chars_q_raw, which is just a temporary dataframe       #
########################################################################################################################

with open('chars_q_accounting.feather', 'rb') as f:
    chars_q = feather.read_feather(f)
f.close()

chars_q = chars_q.dropna(subset=['permno'])
chars_q[['permno', 'gvkey']] = chars_q[['permno', 'gvkey']].astype(int)
chars_q['jdate'] = pd.to_datetime(chars_q['jdate'])
chars_q = chars_q.drop_duplicates(['permno', 'jdate'])

with open('beta.feather', 'rb') as f:
    beta = feather.read_feather(f)
f.close()

beta['permno'] = beta['permno'].astype(int)
beta['jdate'] = pd.to_datetime(beta['date']) + MonthEnd(0)
beta = beta[['permno', 'jdate', 'beta']]
beta = beta.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, beta, how='left', on=['permno', 'jdate'])

with open('rvar_capm.feather', 'rb') as f:
    rvar_capm = feather.read_feather(f)
f.close()

rvar_capm['permno'] = rvar_capm['permno'].astype(int)
rvar_capm['jdate'] = pd.to_datetime(rvar_capm['date']) + MonthEnd(0)
rvar_capm = rvar_capm[['permno', 'jdate', 'rvar_capm']]
rvar_capm = rvar_capm.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, rvar_capm, how='left', on=['permno', 'jdate'])

with open('rvar_mean.feather', 'rb') as f:
    rvar_mean = feather.read_feather(f)
f.close()

rvar_mean['permno'] = rvar_mean['permno'].astype(int)
rvar_mean['jdate'] = pd.to_datetime(rvar_mean['date']) + MonthEnd(0)
rvar_mean = rvar_mean[['permno', 'jdate', 'rvar_mean']]
rvar_mean = rvar_mean.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, rvar_mean, how='left', on=['permno', 'jdate'])

with open('rvar_ff3.feather', 'rb') as f:
    rvar_ff3 = feather.read_feather(f)
f.close()

rvar_ff3['permno'] = rvar_ff3['permno'].astype(int)
rvar_ff3['jdate'] = pd.to_datetime(rvar_ff3['date']) + MonthEnd(0)
rvar_ff3 = rvar_ff3[['permno', 'jdate', 'rvar_ff3']]
rvar_ff3 = rvar_ff3.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, rvar_ff3, how='left', on=['permno', 'jdate'])

with open('sue.feather', 'rb') as f:
    sue = feather.read_feather(f)
f.close()

sue['permno'] = sue['permno'].astype(int)
sue['jdate'] = pd.to_datetime(sue['date']) + MonthEnd(0)
sue = sue[['permno', 'jdate', 'sue']]
sue = sue.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, sue, how='left', on=['permno', 'jdate'])

with open('myre.feather', 'rb') as f:
    re = feather.read_feather(f)
f.close()

re['permno'] = re['permno'].astype(int)
re['jdate'] = pd.to_datetime(re['date']) + MonthEnd(0)
re = re[['permno', 'jdate', 're']]
re = re.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, re, how='left', on=['permno', 'jdate'])

with open('abr.feather', 'rb') as f:
    abr = feather.read_feather(f)
f.close()

abr['permno'] = abr['permno'].astype(int)
abr['jdate'] = pd.to_datetime(abr['date']) + MonthEnd(0)
abr = abr[['permno', 'jdate', 'abr']]
abr = abr.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, abr, how='left', on=['permno', 'jdate'])

with open('baspread.feather', 'rb') as f:
    baspread = feather.read_feather(f)
f.close()

baspread['permno'] = baspread['permno'].astype(int)
baspread['jdate'] = pd.to_datetime(baspread['date']) + MonthEnd(0)
baspread = baspread[['permno', 'jdate', 'baspread']]
baspread = baspread.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, baspread, how='left', on=['permno', 'jdate'])

with open('maxret.feather', 'rb') as f:
    maxret = feather.read_feather(f)
f.close()

maxret['permno'] = maxret['permno'].astype(int)
maxret['jdate'] = pd.to_datetime(maxret['date']) + MonthEnd(0)
maxret = maxret[['permno', 'jdate', 'maxret']]
maxret = maxret.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, maxret, how='left', on=['permno', 'jdate'])

with open('std_dolvol.feather', 'rb') as f:
    std_dolvol = feather.read_feather(f)
f.close()

std_dolvol['permno'] = std_dolvol['permno'].astype(int)
std_dolvol['jdate'] = pd.to_datetime(std_dolvol['date']) + MonthEnd(0)
std_dolvol = std_dolvol[['permno', 'jdate', 'std_dolvol']]
std_dolvol = std_dolvol.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, std_dolvol, how='left', on=['permno', 'jdate'])

with open('ill.feather', 'rb') as f:
    ill = feather.read_feather(f)
f.close()

ill['permno'] = ill['permno'].astype(int)
ill['jdate'] = pd.to_datetime(ill['date']) + MonthEnd(0)
ill = ill[['permno', 'jdate', 'ill']]
ill = ill.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, ill, how='left', on=['permno', 'jdate'])

with open('std_turn.feather', 'rb') as f:
    std_turn = feather.read_feather(f)
f.close()

std_turn['permno'] = std_turn['permno'].astype(int)
std_turn['jdate'] = pd.to_datetime(std_turn['date']) + MonthEnd(0)
std_turn = std_turn[['permno', 'jdate', 'std_turn']]
std_turn = std_turn.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, std_turn, how='left', on=['permno', 'jdate'])

with open('zerotrade.feather', 'rb') as f:
    zerotrade = feather.read_feather(f)
f.close()

zerotrade['permno'] = zerotrade['permno'].astype(int)
zerotrade['jdate'] = pd.to_datetime(zerotrade['date']) + MonthEnd(0)
zerotrade = zerotrade[['permno', 'jdate', 'zerotrade']]
zerotrade = zerotrade.drop_duplicates(['permno', 'jdate'])

chars_q = pd.merge(chars_q, zerotrade, how='left', on=['permno', 'jdate'])

# fill the return
chars_q = pd.merge(chars_q, crsp, how='left', on=['permno', 'jdate'])
chars_q['ret'] = np.where(chars_q['ret'].isnull(), chars_q['ret_fill'], chars_q['ret'])
chars_q['retx'] = np.where(chars_q['retx'].isnull(), chars_q['retx_fill'], chars_q['retx'])
chars_q['me'] = np.where(chars_q['me'].isnull(), chars_q['me_fill'], chars_q['me'])
# chars_q['exchcd'] = np.where(chars_q['exchcd'].isnull(), chars_q['exchcd_fill'], chars_q['exchcd'])
# chars_q['shrcd'] = np.where(chars_q['shrcd'].isnull(), chars_q['shrcd_fill'], chars_q['shrcd'])

chars_q = chars_q.dropna(subset=['permno', 'jdate', 'ret', 'retx'])
# chars_q = chars_q[((chars_q['exchcd'] == 1) | (chars_q['exchcd'] == 2) | (chars_q['exchcd'] == 3)) &
#                    ((chars_q['shrcd'] == 10) | (chars_q['shrcd'] == 11))]
chars_q = chars_q.loc[(chars_q.primaryexch.isin(['N', 'A', 'Q'])) & \
                   (chars_q.conditionaltype =='RW') & \
                   (chars_q.tradingstatusflg =='A')]
# equivalent to legacy code shrcd = 10 or 11
chars_q = chars_q.loc[(chars_q.sharetype=='NS') & \
                    (chars_q.securitytype=='EQTY') & \
                    (chars_q.securitysubtype=='COM') & \
                    (chars_q.usincflg=='Y') & \
                    (chars_q.issuertype.isin(['ACOR', 'CORP']))]

# save data
with open('chars_q_raw.feather', 'wb') as f:
    feather.write_feather(chars_q, f)
f.close()

conn.close()
