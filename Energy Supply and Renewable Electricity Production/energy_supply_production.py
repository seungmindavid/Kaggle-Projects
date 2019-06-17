import pandas as pd
import numpy as np

def answer_one():
    energy = pd.read_excel('Energy Indicators.xls', 'Energy', skiprows = 17, usecols= [2,3,4,5,6], skip_footer=38)
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy = energy.replace("...", value= np.NaN)
    energy['Energy Supply'] *= 1000000

    energy = energy.replace(regex= True, to_replace=[r'\s\(([^*)]+)\)', r'\d'], value=r'')
    energy = energy.replace({'Country': {"Republic of Korea": "South Korea", "United States of America": "United States",
                                    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                    "China, Hong Kong Special Administrative Region": "Hong Kong"}})

    gdp = pd.read_csv('world_bank.csv', skiprows=4)
    gdp = gdp.replace({'Country Name': {"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong"}})

    scimEn = pd.read_excel('scimagojr-3.xlsx')
    scimEn = scimEn.set_index('Country')
    energy = energy.set_index('Country')
    gdp = gdp.set_index('Country Name')
    gdp = gdp.loc[:, '2006':'2015']

    scimEn = scimEn[scimEn['Rank'] < 16]
    ans = pd.merge(scimEn, energy, how= 'inner',left_index=True, right_index=True)
    ans = pd.merge(ans, gdp, how='inner', left_index=True, right_index=True)
    ans
    return ans

answer_one()


def answer_two():
    energy = pd.read_excel('Energy Indicators.xls', 'Energy', skiprows = 17, usecols= [2,3,4,5,6], skip_footer=38)
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy = energy.replace("...", value= np.NaN)
    energy['Energy Supply'] *= 1000000

    energy = energy.replace(regex= True, to_replace=[r'\s\(([^*)]+)\)', r'\d'], value=r'')
    energy = energy.replace({'Country': {"Republic of Korea": "South Korea", "United States of America": "United States",
                                    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                    "China, Hong Kong Special Administrative Region": "Hong Kong"}})

    gdp = pd.read_csv('world_bank.csv', skiprows=4)
    gdp = gdp.replace({'Country Name': {"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong"}})

    scimEn = pd.read_excel('scimagojr-3.xlsx')
    scimEn = scimEn.set_index('Country')
    energy = energy.set_index('Country')
    gdp = gdp.set_index('Country Name')
    gdp = gdp.loc[:, '2006':'2015']

    energy_len = len(energy)
    gdp_len = len(gdp)
    scimEn_len = len(scimEn)
    
    e_gdp = pd.merge(energy, gdp, how= 'inner',left_index=True, right_index=True)
    e_scim = pd.merge(energy, scimEn, how= 'inner',left_index=True, right_index=True)
    g_scim = pd.merge(gdp, scimEn, how= 'inner',left_index=True, right_index=True)
    e_g_scim = pd.merge(e_gdp, scimEn, how= 'inner',left_index=True, right_index=True)
    
    total = energy_len + gdp_len + scimEn_len + len(e_g_scim) - len(e_gdp) - len(e_scim) - len(g_scim)
    ans = total - len(e_g_scim)
    #print(energy_len)
    #print(gdp_len)
    #print(scimEn_len)
    #print(len(e_g_scim))
    #print(len(e_gdp))
    #print(len(e_scim))
    #print(len(g_scim))
    return ans

def answer_three():
    df = answer_one()
    df.head()
    df['avgGDP'] = df.loc[:,'2006':'2015'].T.mean()
    ans = df['avgGDP']
    ans = ans.sort_values(ascending = False)
    return ans


def answer_four():
    df = answer_one()
    avg_gdp = answer_three()
    sixth = avg_gdp.keys()[5]
    ans = df.loc[sixth]['2015'] - df.loc[sixth]['2006']
    return ans


def answer_five():
    df = answer_one()
    ans = df['Energy Supply per Capita'].mean()
    return ans

def answer_six():
    df = answer_one()
    maximum = df[df['% Renewable'] == df['% Renewable'].max()]
    name = maximum.index[0]
    perc = maximum.loc[name]['% Renewable']
    ans = (name, perc)
    return ans


def answer_seven():
    df = answer_one()
    df['Citation Ratio'] = df['Self-citations'] / df['Citations']
    maximum = df[df['Citation Ratio'] == df['Citation Ratio'].max()]
    ans = (maximum.index[0], maximum.loc[maximum.index[0]]['Citation Ratio'])
    return ans

def answer_eight():
    df = answer_one()
    df['population'] = df['Energy Supply'] / df['Energy Supply per Capita']
    popInSort = df['population'].sort_values(ascending=False)
    ans = popInSort.keys()[2]
    return ans


def answer_nine():
    df = answer_one()
    df['capita'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df['Citation Documents per Capita'] = df['Citable documents'] / df['capita']
    ans = df[['Energy Supply per Capita','Citation Documents per Capita']]
    ans = ans.corr(method= 'pearson')
    #print(ans.loc['Energy Supply per Capita']['Citation Documents per Capita'])
    return ans.loc['Energy Supply per Capita']['Citation Documents per Capita']

def answer_ten():
    df = answer_one()
    df['HighRenew'] = 0
    df['% Renewable'].median()
    df['HighRenew'] = df['% Renewable'] > df['% Renewable'].median()

    df = df.sort_values(by= ['Rank'], ascending=True)
    df['HighRenew'] = df['HighRenew']*1
    return df['HighRenew']

def answer_eleven():
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    df = answer_one()
    df['capita'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df['Continent'] = None
    for country in df.index:
        df['Continent'][country] = ContinentDict[country]

    df['Continent']
    ans = df.groupby('Continent')['capita'].agg(['size', 'sum', 'mean', 'std'])
    return ans

def answer_twelve():
    df = answer_one()
    df['capita'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df['Continent'] = None
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    
    for country in df.index:
        df['Continent'][country] = ContinentDict[country]
    df['% Renewable into 5 Bin'] = pd.cut(df['% Renewable'],5)
    df = df.reset_index()
    df = df.set_index(['Continent', '% Renewable into 5 Bin'])
    ans = df.groupby(level=['Continent', '% Renewable into 5 Bin']).size()
    return ans

answer_twelve()

def answer_thirteen():
    df = answer_one()
    df['PopEst'] = df['Energy Supply'] / df['Energy Supply per Capita']
    df['PopEst'] = df['PopEst'].apply(lambda x:format(x,','))
    ans = df['PopEst']
    return ans




