import pandas as pd
import numpy as np

#import data
df = pd.read_csv("./_data/github_jacksonjude/2000-2020-countypres.csv")

# filters anything after 2020 since our previous df contains that data
df_76 = pd.read_csv("./_data/github_jacksonjude/1976-2020-president.csv")
df_76 = df_76[df_76['year'] < 2000]

electoral_votes_df = pd.read_csv("./_data/electoral_data.csv")

# FRED DATA
# Import CPI Data (from FRED data)
df_cpi = pd.read_csv('./_data/FPCPITOTLZGUSA.csv')
df_cpi['year'] = pd.to_datetime(df_cpi['DATE']).dt.year
df_cpi.rename(columns={'FPCPITOTLZGUSA': 'cpi'}, inplace=True)
df_cpi = df_cpi.drop(columns=['DATE'])
print(df_cpi.head())

#Import Uber Election Data
df_u = pd.read_csv('./_data/UNRATE.csv', header=1, names=['DATE', 'UNRATE'])
# df = pd.read_csv('your_file.csv', header=None, names=['date', 'unemployment_rate'])
df_u['year'] = pd.to_datetime(df_u['DATE'])
df_unemployment = df_u[df_u['year'].dt.month == 9].reset_index(drop=True)
df_unemployment['year'] = df_unemployment['year'].dt.year
df_unemployment = df_unemployment.drop(columns=['DATE'])
df_unemployment.rename(columns={'UNRATE': 'unemployment_rate'}, inplace=True)
df_unemployment = df_unemployment[df_unemployment['year'] >= 1976]
print(df_unemployment.head())

# GDP Growth rate
df_gdp_growth = pd.read_csv('./_data/united-states-gdp-growth-rate.csv')
df_gdp_growth['Date'] = pd.to_datetime(df_gdp_growth['Date'])
df_gdp_growth['year'] = df_gdp_growth['Date'].dt.year
df_gdp_growth = df_gdp_growth.drop(columns=['Date'])
print(df_gdp_growth.head())


# Sanitice Data and Join the two dataframes

# First, group by year, state, and candidate to sum the votes at the county level
df['candidate'] = df['candidate'].str.title()
df['state'] = df['state'].str.title()
df['party'] = df['party'].str.title()
df.drop(columns=['office','state_po','county_fips','version','mode','county_name'], inplace=True)

df['state'] = df['state'].str.strip().str.title()
electoral_votes_df['state'] = electoral_votes_df['state'].str.strip().str.title()

# Split the 'name' column into 'Last' and 'First'
def reverse_name(name):
    if isinstance(name, str):  # Check if the value is a string
        parts = name.replace(',', '').split()
        if len(parts) == 2:  # Ensure it has both first and last names
            return f"{parts[1]} {parts[0]}"
    return name  # Return the original if not a string or not in expected format

# Apply the function to the 'candidate' column
df_76['formatted_candidate'] = df_76['candidate'].apply(reverse_name).str.title()
df_76.drop(columns=['office','state_fips','state_cen','state_po', 'state_ic','version', 'writein', 'notes','party_simplified','candidate'], inplace=True)
df_76.rename(columns={'formatted_candidate': 'candidate'}, inplace=True)
df_76.rename(columns={'party_detailed': 'party'}, inplace=True)
df_76['state'] = df_76['state'].str.title()
df_76['party'] = df_76['party'].str.title()

df_all = pd.concat([df_76, df], axis=0, ignore_index=True)
# Replace NaN with 0
df_all.fillna(0, inplace=True)
df_all['candidatevotes'] = df_all['candidatevotes'].fillna(0)
df_all['totalvotes'] = df_all['totalvotes'].fillna(0)


# Tabulate the results
def get_state_winning_party(year, state):
    state_year_df = df_all[(df_all['year'] == year) & (df_all['state'] == state)]

    dem_state_votes = state_year_df[state_year_df['party'] == 'Democrat']['candidatevotes'].sum()
    rep_state_votes = state_year_df[state_year_df['party'] == 'Republican']['candidatevotes'].sum()

    if dem_state_votes > rep_state_votes:
        dem_electoral_votes = electoral_votes_df[electoral_votes_df['state'] == state].sum()['electoral_votes']
        rep_electoral_votes = 0
    else:
        dem_electoral_votes = 0
        rep_electoral_votes = electoral_votes_df[electoral_votes_df['state'] == state].sum()['electoral_votes']
    return dem_electoral_votes, rep_electoral_votes

def build_dataframe_data_for_year(year):
    # Filter the DataFrame for the specified year
    filtered_df = df_all[df_all['year'] == year]
    dem_electoral_votes = 0
    rep_electoral_votes = 0

    # create dataframe with year, state, candidate, electoral votes, winner    
    resultsByState = pd.DataFrame()

    for state in filtered_df['state'].unique():
        dem_electoral_state_votes, rep_electoral_states_votes = get_state_winning_party(year, state)
        dem_electoral_votes += dem_electoral_state_votes
        rep_electoral_votes += rep_electoral_states_votes
        total_state_votes = filtered_df[(filtered_df['year'] == year) & (filtered_df['state'] == state)]['candidatevotes'].sum()

        for candidate in filtered_df[(filtered_df['year'] == year) & (filtered_df['state'] == state)]['candidate'].unique():
            
            if candidate != 0:
                party = filtered_df[(filtered_df['year'] == year) & (filtered_df['state'] == state) & (filtered_df['candidate'] == candidate)]['party'].unique()[0]
                if ((party == 'Democrat') | (party == 'Republican')):
                    if (dem_electoral_state_votes > rep_electoral_states_votes):
                        winner = 'Democrat'
                        if party == 'Democrat':
                            electoral_votes = dem_electoral_state_votes
                        else:
                            electoral_votes = 0
                    else:
                        winner = 'Republican'
                        if party == 'Republican':
                            electoral_votes = rep_electoral_states_votes
                        else:
                            electoral_votes = 0

                    candidate_votes = filtered_df[(filtered_df['year'] == year) & (filtered_df['state'] == state) & (filtered_df['candidate'] == candidate)]['candidatevotes'].sum()

                    # print(f"{year}: {state} {candidate} ({party}) got {candidate_votes} votes and {electoral_votes} electoral votes.  State winner: {winner}")

                    state_df = pd.DataFrame({
                        'year': [year],
                        'state': [state],
                        'candidate': [candidate],
                        'party': [party],
                        'electoral_votes': [electoral_votes],
                        'candidate_votes': [candidate_votes],
                        'total_state_votes': [total_state_votes],
                        'state_winner': [winner]
                        })
                    # print(state_df)
                    resultsByState = pd.concat([resultsByState, state_df], ignore_index=True)
                    # print(state_df)
    
    # print(f"{year}: dem electoral {dem_electoral_votes} rep electoral {rep_electoral_votes}")
    if (dem_electoral_votes > rep_electoral_votes):
        # add a winner column
        resultsByState['winner'] = 'Democrat'
    else:
        resultsByState['winner'] = 'Republican'

    # add cpi to results
    cpi = df_cpi[df_cpi['year'] == year]['cpi'].iloc[0]
    resultsByState['cpi'] = cpi
    
    # df_unemployment = df_unemployment[df_unemployment['year'] == year]
    # resultsByState['unemployment_rate'] = df_unemployment

    if (df_unemployment['year'] == year).any():
        # If the year exists, filter the DataFrame and assign the unemployment rate
        unemployment_rate = df_unemployment.loc[df_unemployment['year'] == year, 'unemployment_rate'].values[0]
        resultsByState['unemployment_rate'] = unemployment_rate
    else:
        # Handle the case where the year does not exist
        resultsByState['unemployment_rate'] = None  # Or some other default value

    # if (df_gdp_growth['year'] == year).any():
    #     gdp_growth = df_gdp_growth.loc[df_gdp_growth['year'] == year, 'gdp_ptc_change'].values[0]
    #     resultsByState['gdp_ptc_change'] = gdp_growth

    return resultsByState
    


# Build Dataframe to use with pytorch
election_results_df = pd.DataFrame()

for year in range(1976, 2020, 4):
    results = build_dataframe_data_for_year(year)
    election_results_df = pd.concat([election_results_df, results], ignore_index=True)
    print(results)

print(election_results_df.dtypes)
