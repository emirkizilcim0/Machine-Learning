import pandas as pd

# Creating a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})

# Shuffle the DataFrame
shuffled_df = df.sample(frac=1)

print(shuffled_df)
