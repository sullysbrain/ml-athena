import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

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

#Import Uber Election Data
df_u = pd.read_csv('./_data/UNRATE.csv', header=1, names=['DATE', 'UNRATE'])
# df = pd.read_csv('your_file.csv', header=None, names=['date', 'unemployment_rate'])
df_u['year'] = pd.to_datetime(df_u['DATE'])
df_unemployment = df_u[df_u['year'].dt.month == 9].reset_index(drop=True)
df_unemployment['year'] = df_unemployment['year'].dt.year
df_unemployment = df_unemployment.drop(columns=['DATE'])
df_unemployment.rename(columns={'UNRATE': 'unemployment_rate'}, inplace=True)
df_unemployment = df_unemployment[df_unemployment['year'] >= 1976]

# GDP Growth rate
df_gdp_growth = pd.read_csv('./_data/united-states-gdp-growth-rate.csv')
df_gdp_growth['Date'] = pd.to_datetime(df_gdp_growth['Date'])
df_gdp_growth['year'] = df_gdp_growth['Date'].dt.year
df_gdp_growth = df_gdp_growth.drop(columns=['Date'])

df_job_approval = pd.read_csv('./_data/incumbant_job_approval.csv')
df_job_approval['year'] = df_job_approval['year'].astype(int)

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

    ## TEMP FIX FOR FLORIDA 2000 recound and data mismatch
    if year == 2000 and state == 'Florida':
        rep_state_votes = 2912790
        dem_state_votes = 2912253

    if dem_state_votes > rep_state_votes:
        dem_electoral_votes = electoral_votes_df[electoral_votes_df['state'] == state].sum()['electoral_votes']
        rep_electoral_votes = 0
    else:
        dem_electoral_votes = 0
        rep_electoral_votes = electoral_votes_df[electoral_votes_df['state'] == state].sum()['electoral_votes']

    # print(f"{year} {state} r: {rep_state_votes:,} d: {dem_state_votes:,} {rep_electoral_votes}")
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
        resultsByState['winner'] = 0   # Democrat = 0
    else:
        resultsByState['winner'] = 1  # Republican=  1

    resultsByState['dem_electoral_total'] = dem_electoral_votes
    resultsByState['rep_electoral_total'] = rep_electoral_votes


    # add cpi to results
    cpi = df_cpi[df_cpi['year'] == year]['cpi'].iloc[0]
    resultsByState['cpi'] = cpi
    
    if (df_unemployment['year'] == year).any():
        # If the year exists, filter the DataFrame and assign the unemployment rate
        unemployment_rate = df_unemployment.loc[df_unemployment['year'] == year, 'unemployment_rate'].values[0]
        resultsByState['unemployment_rate'] = unemployment_rate
    else:
        # Handle the case where the year does not exist
        resultsByState['unemployment_rate'] = None  # Or some other default value

    gdp_growth_year = df_gdp_growth[df_gdp_growth['year'] == year]
    resultsByState['gdp_ptc_change'] = gdp_growth_year.values[0][1]

    return resultsByState
    


# Build Dataframe to use with pytorch
election_results_df = pd.DataFrame()

# print("Historical data for training....")
starting_year = 1976
ending_year = 2024

for year in range(starting_year, ending_year, 4):
    results = build_dataframe_data_for_year(year)
    election_results_df = pd.concat([election_results_df, results], ignore_index=True)

# Add incumbent data
for year in range(starting_year, ending_year, 4):
    if year == starting_year:
        incumbent_party = 1
    else:
        previous_election_year = year - 4
        incumbent_party = election_results_df[election_results_df['year'] == previous_election_year]['winner'].iloc[0]
    
    job_approval = df_job_approval[df_job_approval['year'] == year]
    job_approval_val = job_approval.values[0][1]

    election_results_df.loc[election_results_df['year'] == year, 'pres_job_approval'] = job_approval_val
    election_results_df.loc[election_results_df['year'] == year, 'incumbent_party'] = incumbent_party



features = election_results_df[[
    'year', 
    'cpi', 
    'unemployment_rate', 
    'gdp_ptc_change',
    'pres_job_approval',
    ]]

targets = election_results_df[[
    'dem_electoral_total', 
    'rep_electoral_total', 
    ]]


# Convert DataFrame to numpy arrays
X = features.values
y = targets[['dem_electoral_total', 'rep_electoral_total']].values  # Change as needed


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)



class ElectionModel(nn.Module):
    def __init__(self, input_size):
        super(ElectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)           # Second hidden layer
        self.fc3 = nn.Linear(64, 2)            # Output layer (two targets)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model
input_size = X_train.shape[1]  # Number of features
model = ElectionModel(input_size)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Train the model
best_model_state = None
best_loss = float('inf')
num_epochs = 50000

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    # Check if current loss is the best
    if loss < best_loss:
        best_loss = loss
        best_model_state = model.state_dict()  # Save model state

    if (epoch + 1) % 5000 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# After training, load the best model
print(f"Best Loss: {best_loss.item():.4f}")
model.load_state_dict(best_model_state)



## Predict
## Test data: 2020
## Actual:  Dem: 306  Rep: 232   Dem win
# new_data = {
#     'year': [2020],
#     'cpi': [1.2],
#     'unemployment_rate': [6.8],
#     'gdp_ptc_change': [-4],
#     # 'incumbent_party': [1],
#     'pres_job_approval': [43]
# }
# 2024
new_data = {
    'year': [2024],
    'cpi': [2.4],
    'unemployment_rate': [4.1],
    'gdp_ptc_change': [3],
    # 'incumbent_party': [0],
    'pres_job_approval': [35]
}

new_data_df = pd.DataFrame(new_data)
test_tensor = torch.tensor(new_data_df.values, dtype=torch.float32)  


model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No gradient calculation needed
    test_outputs = model(test_tensor)


predicted_dem_electoral_votes = test_outputs[:, 0]
predicted_rep_electoral_votes = test_outputs[:, 1]
# predicted_winner = test_outputs[:, 2]
# winner = "Democrat" if predicted_winner[0] == 0 else "Republican"


print(f"\nPredictions for {new_data_df['year'][0]}:")
print(f"Dem: {predicted_dem_electoral_votes[0]:.0f} electoral votes\nRep: {predicted_rep_electoral_votes[0]:.0f} electoral votes")


