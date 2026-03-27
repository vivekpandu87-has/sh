
import pandas as pd
import numpy as np
import random

np.random.seed(42)
n = 2500
data = {
    'Respondent_ID': [f'RESP_{i:04d}' for i in range(n)],
    'Age_Group': np.random.choice(['18-24', '25-34', '35+'], n, p=[0.45, 0.4, 0.15]),
    'City_Tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n, p=[0.5, 0.3, 0.2]),
    'Occupation': np.random.choice(['Student', 'Corporate', 'Creative', 'Other'], n),
    'Monthly_Reading_Hours': np.random.poisson(10, n),
    'Primary_Format': np.random.choice(['Physical Books', 'E-books/Kindle', 'Audiobooks', 'Hybrid'], n),
    'Reading_Motivation': np.random.choice(['Learn', 'Escape', 'Status', 'Detox'], n),
    'Attention_Span': np.random.choice(['<15m', '15-30m', '30-60m', '1hr+'], n),
    'Aesthetic_Importance': np.random.randint(1, 6, n),
    'Comp_Tea_Coffee': np.random.randint(0, 2, n),
    'Comp_Scented_Candle': np.random.randint(0, 2, n),
    'Comp_Music': np.random.randint(0, 2, n),
    'Comp_Snacks': np.random.randint(0, 2, n),
    'Inv_Kindle': np.random.randint(0, 2, n),
    'Inv_Reading_Light': np.random.randint(0, 2, n),
    'Inv_Bookmarks': np.random.randint(0, 2, n),
    'Inv_Scented_Candles': np.random.randint(0, 2, n),
    'Inv_Planner': np.random.randint(0, 2, n),
    'Monthly_Lifestyle_Spend': np.random.normal(15000, 5000, n).astype(int),
    'Eco_Importance': np.random.randint(1, 6, n),
    'Final_Intent': np.random.choice(['Yes', 'No', 'Maybe'], n)
}
df = pd.DataFrame(data)
df.to_csv('reading_renaissance_data.csv', index=False)
print("Dataset Created!")
