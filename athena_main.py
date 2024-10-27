import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#region Import Data

# Import data
def import_raw_data():
    df = pd.read_csv("./_data/github_jacksonjude/2000-2020-countypres.csv")

    # filters anything after 2020 since our previous df contains that data
    df_76 = pd.read_csv("./_data/github_jacksonjude/1976-2020-president.csv")
    df_76 = df_76[df_76['year'] < 2000]

    # First, group by year, state, and candidate to sum the votes at the county level
    df['state'] = df['state'].str.strip().str.title()
    df['party'] = df['party'].str.title()
    df.drop(columns=['office','state_po','county_fips','version','mode','county_name', 'candidate'], inplace=True)

    df_76.drop(columns=['office','state_fips','state_cen','state_po', 'state_ic','version', 'writein', 'notes','party_simplified','candidate'], inplace=True)
    df_76.rename(columns={'party_detailed': 'party'}, inplace=True)
    df_76['state'] = df_76['state'].str.title()
    df_76['party'] = df_76['party'].str.title()

    df_all = pd.concat([df_76, df], axis=0, ignore_index=True)
    # Replace NaN with 0
    df_all.fillna(0, inplace=True)
    df_all['totalvotes'] = df_all['totalvotes'].fillna(0)

    df_filtered_dem_rep = df_all[df_all['party'].isin(['Democrat', 'Republican'])]

    return df_filtered_dem_rep

# Add Macroeconomic Data
def load_cpi():
    df_cpi = pd.read_csv('./_data/FPCPITOTLZGUSA.csv')
    df_cpi['year'] = pd.to_datetime(df_cpi['DATE']).dt.year
    df_cpi.rename(columns={'FPCPITOTLZGUSA': 'cpi'}, inplace=True)
    df_cpi = df_cpi.drop(columns=['DATE'])
    return df_cpi

def load_unemployment():
    df_unemployment = pd.read_csv('./_data/UNRATE.csv', header=1, names=['DATE', 'UNRATE'])
    # df = pd.read_csv('your_file.csv', header=None, names=['date', 'unemployment_rate'])
    df_unemployment['year'] = pd.to_datetime(df_unemployment['DATE'])
    df_unemployment = df_unemployment[df_unemployment['year'].dt.month == 9].reset_index(drop=True)
    df_unemployment['year'] = df_unemployment['year'].dt.year
    df_unemployment = df_unemployment.drop(columns=['DATE'])
    df_unemployment.rename(columns={'UNRATE': 'unemployment_rate'}, inplace=True)
    df_unemployment = df_unemployment[df_unemployment['year'] >= 1976]
    return df_unemployment

def load_gdp_growth():
    df_gdp_growth = pd.read_csv('./_data/united-states-gdp-growth-rate.csv')
    df_gdp_growth['Date'] = pd.to_datetime(df_gdp_growth['Date'])
    df_gdp_growth['year'] = df_gdp_growth['Date'].dt.year
    df_gdp_growth = df_gdp_growth.drop(columns=['Date'])
    return df_gdp_growth

def load_president_job_approval():
    df_job_approval = pd.read_csv('./_data/incumbant_job_approval.csv')
    # df_job_approval['year'] = df_job_approval['year'].astype(int)
    # print(f"\n\nLoaded job approval.. \n{df_job_approval.head()}")
    return df_job_approval

# Prep Data
df_electionData = import_raw_data()
df_cpi = load_cpi()
df_unemployment = load_unemployment()
df_gdp_growth = load_gdp_growth()
df_job_approval = load_president_job_approval()
# Load States
states = df_electionData['state'].unique()


#endregion

#region Format Election Data
# format election historic data
def get_state_winning_party(year, state):
    state_year_df = df_electionData[(df_electionData['year'] == year) & (df_electionData['state'] == state)]

    dem_state_votes = state_year_df[state_year_df['party'] == 'Democrat']['candidatevotes'].sum()
    rep_state_votes = state_year_df[state_year_df['party'] == 'Republican']['candidatevotes'].sum()
    total_state_votes = state_year_df[state_year_df['party'] == 'Republican']['totalvotes'].sum()

    ## TEMP FIX FOR FLORIDA 2000 recound and data mismatch
    if year == 2000 and state == 'Florida':
        rep_state_votes = 2912790
        dem_state_votes = 2912253

    if dem_state_votes > rep_state_votes:
        winner = "Democrat"
    else:  
        winner = "Republican"
    return dem_state_votes, rep_state_votes, total_state_votes

# Constants
start_year = 1976
end_year = 2024

fullFeature_rows = []
for state in states:
    for year in range(start_year, end_year, 4):
        year_row = []
        dem_votes, rep_votes, total_votes = get_state_winning_party(year, state)
        if dem_votes > rep_votes:
            winner = "Democrat"
            winning_part_numeric = -1
        else:  
            winner = "Republican"
            winning_part_numeric = 1

        # Inflation for last 2 years
        if start_year == year:
            cpi_previous = 0
        else:
            cpi_previous = df_cpi[df_cpi['year'] == (year-1)]['cpi'].values[0]

        cpi = df_cpi[df_cpi['year'] == year]['cpi'].values[0] + cpi_previous
        gdp = df_gdp_growth.loc[df_gdp_growth['year'] == year].reset_index(drop=True).iloc[0, 0]
        unemployment = df_unemployment[df_unemployment['year'] == year].reset_index(drop=True).iloc[0,0]
        job_approval = df_job_approval[df_job_approval['year'] == year].reset_index(drop=True).iloc[0,1]
        last_dem_poll = df_job_approval[df_job_approval['year'] == year].reset_index(drop=True).iloc[0,2]
        last_rep_poll = df_job_approval[df_job_approval['year'] == year].reset_index(drop=True).iloc[0,3]
        incumbant_wins = df_job_approval[df_job_approval['year'] == year].reset_index(drop=True).iloc[0,4]
        incumbant_party = df_job_approval[df_job_approval['year'] == year].reset_index(drop=True).iloc[0,5]

        state_votes = []
        state_votes.append({
            'year': year,
            'state': state,
            'cpi': cpi,
            'gdp': gdp,
            'unemployment_rate': unemployment,
            'job_approval': job_approval,
            'last_dem_poll': last_dem_poll,
            'last_rep_poll': last_rep_poll,
            'incumbant_wins': incumbant_wins,
            'incumbant_party': incumbant_party,
            'dem_votes': dem_votes,
            'rep_votes': rep_votes,
            'total_state_votes': total_votes,
            }) 
        fullFeature_rows.append(state_votes[0])


# SAVE FULL DATAFRAME
df_fullFeatures = pd.DataFrame(fullFeature_rows)

# Normalize Data
features_to_normalize = ['dem_votes', 'rep_votes','total_state_votes']

def normalize_data(df):
    df_normalized_raw = df[features_to_normalize].copy()
#     # normalized_df = (df_normalized_raw - df_normalized_raw.mean()) / df_normalized_raw.std()
    normalized_df = (df_normalized_raw - df_normalized_raw.min()) / (df_normalized_raw.max() - df_normalized_raw.min())
    return normalized_df


# print("\n\nNormalizing Data...\n")
# normalized_dataframe = normalize_data(df_fullFeatures) 
# df_fullFeatures.loc[:, features_to_normalize] = normalized_dataframe[features_to_normalize]


# Create a mapping of state names to indices for use in the Embedding layer
df_fullFeatures['state_idx'] = pd.factorize(df_fullFeatures['state'])[0]

# Create a mapping DataFrame to view state and its corresponding index
state_mapping = df_fullFeatures[['state', 'state_idx']].drop_duplicates().reset_index(drop=True)


# Step 1: Extract state indices as a tensor
state_indices = torch.tensor(df_fullFeatures['state_idx'].values, dtype=torch.long)


# Step 2: Select numeric features for the input tensor
# Assuming these are the relevant numeric features (you can adjust as needed)
# numeric_features = df_fullFeatures[['cpi', 'gdp', 'unemployment_rate', 
#                                      'job_approval', 'last_dem_poll', 
#                                      'last_rep_poll', 'dem_votes', 'incumbant_wins', 'incumbant_party',
#                                      'rep_votes', 'total_state_votes']]

numeric_features = df_fullFeatures[['cpi', 'gdp', 'unemployment_rate', 
                                     'job_approval', 'last_dem_poll', 
                                     'last_rep_poll', 'dem_votes', 'incumbant_wins', 'incumbant_party',
                                     'rep_votes']]


# Step 3: Convert numeric features to a tensor
numeric_tensor = torch.tensor(numeric_features.values, dtype=torch.float32)




df_training = df_fullFeatures.copy()
#endregion

#region Design Model Class

class ElectionModel(nn.Module):
    def __init__(self, num_states, embedding_dim, num_features):
        super(ElectionModel, self).__init__()
        self.embedding = nn.Embedding(num_states, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + num_features, 64)  # First layer
        self.fc2 = nn.Linear(64, 32)  # Second layer
        self.fc3 = nn.Linear(32, 1)   # Output layer

    def forward(self, state_idx, numeric_features):
        state_embedded = self.embedding(state_idx)
        if numeric_features.shape[1] < self.fc1.in_features - self.embedding.embedding_dim:
            # If less features are provided, pad with zeros or handle accordingly
            padding = torch.zeros(numeric_features.size(0), 
                                  self.fc1.in_features - self.embedding.embedding_dim - 
                                  numeric_features.shape[1]).to(numeric_features.device)
            numeric_features = torch.cat([numeric_features, padding], dim=1)

        x = torch.cat((state_embedded, numeric_features), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Final output
        return x

#endregion

#region Train Model

# Assuming you have your numeric_tensor and state_indices ready
X = numeric_tensor  # Your features
y = df_training['dem_votes'] / df_training['total_state_votes'] * 2 - 1  # Scale votes to [-1, 1]

# Split the data
X_train, X_val, y_train, y_val, state_train, state_val = train_test_split(X, y, state_indices, test_size=0.2, random_state=21)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # Use appropriate dtype
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)  # Use appropriate dtype
y_train = y_train_tensor
y_val = y_val_tensor


# Create datasets
train_dataset = TensorDataset(state_train, X_train, y_train)
val_dataset = TensorDataset(state_val, X_val, y_val)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_states = len(df_fullFeatures['state'].unique())  # Unique states
embedding_dim = 7  # Set as needed
num_features = X.shape[1]  # Number of features in X

print(f"\nEmbedding dim: {embedding_dim}\tNum states: {num_states}\tNum features: {num_features}")

model = ElectionModel(num_states, embedding_dim, num_features)

# Loss functioin definition
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


best_model_state = None
best_loss = float('inf')
num_epochs = 1000  # Adjust based on your needs


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for state_idx, features, target in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(state_idx, features)
 
        # Calculate loss
        loss = criterion(outputs.view(-1), target.float())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print training progress
        # Check if current loss is the best
        if loss < best_loss:
            best_loss = loss
            best_model_state = model.state_dict()  # Save model state

    
    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f'PyTorch Election Model Training: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():,.4f}')


# After training, load the best model
print(f"Best Loss: {best_loss.item():.4f}")
model.load_state_dict(best_model_state)

#endregion


#region Validation
model.eval()  # Set the model to evaluation mode
val_loss = 0
with torch.no_grad():
    for state_idx, features, target in val_loader:
        outputs = model(state_idx, features)
        loss = criterion(outputs.view(-1), target.float())
        val_loss += loss.item()


#endregion

# numeric_features = df_fullFeatures[['cpi', 'gdp', 'unemployment_rate', 
#                                      'job_approval', 'last_dem_poll', 
#                                      'last_rep_poll', 'dem_votes', 
#                                      'rep_votes', 'total_state_votes']]



## Predict
## Test data: 2020
## Actual:  Dem: 306  Rep: 232   Dem win

target_year = 2024
# target_state = 'California'
electoral_votes_dem = 0
electoral_votes_rep = 0
# Type colors
RED = "\033[31m"   # Red color
BLUE = "\033[34m"  # Blue color
RESET = "\033[0m"  # Reset to default color

# Get electoral votes per state
electoral_votes_df = pd.read_csv("./_data/electoral_data.csv")
electoral_votes_df['state'] = electoral_votes_df['state'].str.strip().str.title()


for target_state in states:

    # test_dem_votes = df_fullFeatures[(df_fullFeatures['state'] == target_state) & (df_fullFeatures['year'] == target_year)]['dem_votes']
    # test_dem_votes = df_fullFeatures[(df_fullFeatures['state'] == target_state) & (df_fullFeatures['year'] == target_year)]['dem_votes'].values[0]
    # test_rep_votes = df_fullFeatures[(df_fullFeatures['state'] == target_state) & (df_fullFeatures['year'] == target_year)]['rep_votes'].values[0]
    # test_total_state_votes = df_fullFeatures[(df_fullFeatures['state'] == target_state) & (df_fullFeatures['year'] == target_year)]['total_state_votes'].values[0]

    # print(f"\n\nRetrieved Votes from {target_state}: \nDem: {test_dem_votes:} \nRep: {test_rep_votes:} \nTotal: {test_total_state_votes:}\n\n")

    data_2020 = {
        'cpi': [2.0],
        'gdp': [3.5],
        'unemployment_rate': [4.1], 
        'job_approval': [0.44],
        'last_dem_poll': [0.48],
        'last_rep_poll': [0.48],
        'incumbant_party': [0],
        'incumbant_wins': [0],
        # 'dem_votes': [0],  #test_dem_votes,
        # 'rep_votes': [0],  #test_rep_votes,
        # 'total_state_votes': [0],  #test_total_state_votes
        }

    # State_val is the state mapping index
    X_val_df = pd.DataFrame(data_2020)
    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)  # Use appropriate dtype

    state_idx_val = state_mapping.loc[state_mapping['state'] == target_state, 'state_idx']
    if isinstance(state_idx_val, pd.Series):
        state_idx_val = torch.tensor(state_idx_val.values, dtype=torch.long)  # Use appropriate dtype

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(state_idx_val, X_val)
        predicted_probabilities = predictions.numpy()   # Convert to NumPy array if needed
        probability = predicted_probabilities[0]

    actual_value = probability[0]

    electoral_votes = electoral_votes_df[electoral_votes_df['state'] == target_state]['electoral_votes'].values[0]
    if actual_value < 0.0:
        electoral_votes_dem += electoral_votes
        print(f"{BLUE}{target_state}:\tDem wins {electoral_votes} electoral votes{RESET}")
    else:
        electoral_votes_rep += electoral_votes
        print(f"{RED}{target_state}:\tRep wins {electoral_votes} electoral votes{RESET}")

    # print(f"Predicted probabilities for {target_state}: {probability}")


print(f"\n\nResults:\nDems win {electoral_votes_dem} Electoral Votes\nReps win {electoral_votes_rep} Electoral Votes")
if electoral_votes_dem > electoral_votes_rep:
    print("Dems win")
else:
    print("Reps win")

