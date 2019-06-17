import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map state names to two letter acronyms
#states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota',
#          'VA': 'Virginia'}

university_town_list = []
with open('university_towns.txt') as file:
    for line in file:
        if 'edit' in line:
            state = line.split('[')[0]
        else:
            town = line.split('(')[0]
            university_town_list.append([state, town])

df = pd.DataFrame(university_town_list, columns= ['State', 'RegionName'])
        
def get_gdplev():
    gdp = pd.read_excel('gdplev.xlsx', 'Sheet1', skiprows=7, usecols= [4,5])
    gdp.rename(columns={'Unnamed: 0': 'Quarter', 'Unnamed: 1': 'GDP'}, inplace=True)
    gdp = gdp[212:]
    gdp = gdp.reset_index()
    gdp = gdp.drop(['index'], axis=1)
    return gdp


# Now, we need to get quarter when recession began
# if gdp[0] > gdp[1] and gdp[1] > gdp[2]
def get_recession_start():
    gdp = get_gdplev()
    for i in range(1, len(gdp)-1):
        if (gdp.iloc[i-1].GDP > gdp.iloc[i].GDP) and (gdp.iloc[i].GDP > gdp.iloc[i+1].GDP):
            return gdp.iloc[i-1].Quarter
    return Nonep

# Now, we need to get quarter when recession end
# if gdp[2] < gdp[1] and gdp[2] < gdp[1]
# if gdp[284] < gdp[283] and gdp[283] < gdp[282]

def get_recession_end():
    gdp = get_gdplev()
    recession_start_index = gdp[gdp['Quarter'] ==get_recession_start()].index[0]
    gdp = gdp[recession_start_index:]
    print(gdp)
    for i in range(1, len(gdp)-1):
        if (gdp.iloc[i-1].GDP < gdp.iloc[i].GDP) and (gdp.iloc[i].GDP < gdp.iloc[i+1].GDP):
            return gdp.iloc[i+1].Quarter
    return None

## Now, we need to get quarter when recession is bottom.
# Within a recession, we need to find the lowest gdp
def get_recession_bottom():
    gdp = get_gdplev()
    begin_index = gdp[gdp['Quarter'] == get_recession_start()].index[0]
    end_index= gdp[gdp['Quarter'] == get_recession_end()].index[0]
    gdp_copy = gdp[begin_index:end_index+1]
    bottom_index = gdp_copy[gdp_copy['GDP'] == gdp_copy['GDP'].min()].index[0]
    return gdp.iloc[bottom_index].Quarter


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    allhomes = pd.read_csv('City_Zhvi_AllHomes.csv')
    allhomes['State'] = allhomes['State'].map(states)
    allhomes = allhomes.set_index(['State', 'RegionName'])
    allhomes = allhomes.loc[:,'2001-01':]
    quarters = []
    for i in range(2000,2017):
        for j in ['q1','q2','q3','q4']:
            quarter= str(i) + j
            quarters.append(quarter)
        
    quarters = quarters[:-1]
    mon = 0
    for i in quarters:
        allhomes[i] = allhomes.iloc[:,mon:mon+3].T.mean()
        mon += 3
    allhomes = allhomes.loc[:,'2000q1':]
    allhomes.fillna(np.NaN)
    return allhomes
