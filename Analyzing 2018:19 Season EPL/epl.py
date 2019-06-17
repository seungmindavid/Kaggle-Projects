import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as color

# Create dataframe for goal,assist, throughball, aerial_battle, cleansheet

goal = pd.read_csv('epl_player_goal.csv')
assist = pd.read_csv('epl_player_assist.csv')
throughball = pd.read_csv('epl_player_throughball.csv')
aerial_battle = pd.read_csv('epl_player_aerial_battle.csv')
cleansheet = pd.read_csv('epl_player_cleansheet.csv')
salary = pd.read_csv('epl_player_salary.csv')

# Remove Stat
goal = goal[goal['Goal'] != 'Stat']
assist = assist[assist['Assist'] != 'Stat']
throughball = throughball[throughball['through_ball'] != 'Stat']
aerial_battle = aerial_battle[aerial_battle['Aerial_Battle_Won'] != 'Stat']
cleansheet = cleansheet[cleansheet['Clean_Sheets'] != 'Stat']

# Goal has some duplicate value. We need to drop it
goal = goal.drop_duplicates()

# Merge (Goal, Assist, Throubhall, aerial_battle, cleansheet) with player_name
goal_assist = pd.merge(goal, assist, how ='outer', left_on =['Player_name', 'Club'], right_on =['Player_name', 'Club'])
goal_assist = goal_assist.drop_duplicates()
g_a_through = pd.merge(goal_assist, throughball, how = 'outer', left_on =['Player_name', 'Club'], right_on =['Player_name', 'Club'])
g_a_through = g_a_through.drop_duplicates()
player_stat = pd.merge(g_a_through, aerial_battle, how = 'outer', left_on =['Player_name', 'Club'], right_on =['Player_name', 'Club'])
player_stat = player_stat.drop_duplicates()

# Since Player_stat and Salary data came from different website,
# It has little bit of different format of players' name.
# So I have to import unicodedata to remove all the accent.
# Removing all the accent is the best way to make all the name same

# Since they have different capital letter for the name,
# I will make all the letters to capital
import unicodedata as uni

player_stat['Player_name'] = [uni.normalize('NFKD', x).encode('ASCII', 'ignore') for x in player_stat['Player_name']]
for i in range(len(player_stat['Player_name'])):
    player_stat['Player_name'].iloc[i] = player_stat['Player_name'].iloc[i].decode("utf-8")
    player_stat['Player_name'].iloc[i] = player_stat['Player_name'].iloc[i].upper()

salary['Player_name'] =  [uni.normalize('NFKD', x).encode('ASCII', 'ignore') for x in salary['Player_name']]
for i in range(len(salary['Player_name'])):
    salary['Player_name'].iloc[i] = salary['Player_name'].iloc[i].decode("utf-8")
    salary['Player_name'].iloc[i] = salary['Player_name'].iloc[i].upper()



# Merge Salary and Player_stat
epl_data = pd.merge(player_stat, salary, how='outer', left_on = ['Player_name'], right_on=['Player_name'])
epl_data = epl_data.drop_duplicates()

# Now, let's remove the data which doesn't have team, salary data
# That data doesn't help me to make decision
epl_data = epl_data.dropna(subset=['Player_annual_salary'])
epl_data = epl_data.dropna(subset=['Club'])

# Now, let's fill all the na value with 0
epl_data = epl_data.fillna(0)

# And, remove the £ from the player_annual_salary
# then remove all the ,
epl_data['Player_annual_salary'] = epl_data[epl_data.columns[7:]].replace('[\£,]', '', regex=True).astype(float)
epl_data['Assist'] = pd.to_numeric(epl_data.Assist)
epl_data['Goal'] = pd.to_numeric(epl_data.Goal)
epl_data['through_ball'] = pd.to_numeric(epl_data.through_ball)
epl_data['Aerial_Battle_Won'] = pd.to_numeric(epl_data.Aerial_Battle_Won)

# Midfielder score = Assist *10 + Through_ball * 5+ aerial_battle_won * 3
# Forward score = Goal *10 + assist *5 + through_ball * 3
# Defense score = Aerial_battle_won* 10 + assist * 5 + through_ball * 3
epl_data['mid_score'] = epl_data['Assist']*10 + epl_data['through_ball']*5 + epl_data['Aerial_Battle_Won'] * 3
epl_data['forward_score'] = epl_data['Goal']*10 + epl_data['Assist']*5 + epl_data['through_ball'] * 3
epl_data['defend_score'] = epl_data['Aerial_Battle_Won']*10 + epl_data['Assist']*5 + epl_data['through_ball'] * 3
epl_data['total_score'] = epl_data['mid_score'] + epl_data['forward_score'] + epl_data['defend_score']
# Separate players with their position
defender = epl_data[epl_data['Player_position'] == 'Defender']
midfielder = epl_data[epl_data['Player_position'] == 'Midfielder']
forward = epl_data[epl_data['Player_position'] == 'Forward']

# This number is the mean annual salary per score
annual_salary_per_performance = epl_data['Player_annual_salary'].mean()/epl_data['total_score'].mean()

eplclubs = epl_data['Club'].unique()
club_invested = []
player_performed = []
eplclubs_initial = ['ARS', 'LIV', 'LEI','TOT','MCFC','AFCB', 'BHA', 'MUFC','EFC','BURN','CPFC', 'NUFC',' FFC','WHU','WAT',' SOTON','CHL','CFC','WOLV','HTFC']
for i in eplclubs:
    club_invested.append(epl_data[epl_data['Club'] == i].Player_annual_salary.sum())
    player_performed.append(annual_salary_per_performance * epl_data[epl_data['Club'] == i].total_score.sum())
player_performed_avg = sum(player_performed)/len(player_performed)
print(((player_performed/player_performed_avg) - 1) * 100)

plt.figure(figsize=(15,10))
plt.xticks(range(len(eplclubs_initial)), eplclubs_initial)
plt.xlabel('Clubs')
plt.ylabel('Performance')
plt.title('English Premier League 2018/19 Seasons Investment & Performance')


plt.axhline(y=player_performed_avg, color = 'orange', linestyle = '--', label='Average Performance')

colorlists = color.LinearSegmentedColormap.from_list("colorlists", ["r","w", "b"])
## height/avg
rates = []
for i in range(len(player_performed)):
    r = ((player_performed[i]/player_performed_avg) - 1) * 100
    if r < 0:
        r = 0
    rates.append(r)
# Let's make ScalarMappable which will choose color based on rates
pick_color = cm.ScalarMappable(cmap=colorlists)
pick_color.set_array([])
# Let's update the bar with new colors
plt.bar(range(len(eplclubs_initial)), player_performed, color=pick_color.to_rgba(rates), label = 'Performance')
# Create the colorbar with range 0~50 into 8 pieces
plt.colorbar(pick_color, boundaries = np.linspace(0,50,num =9))
plt.legend()
plt.show()
