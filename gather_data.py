import pandas as pd
import requests
import json
from Detector import evaluate_for_ai
#this contains our generated and real text samples
df = pd.read_csv("calculated.csv")
print(df.head())
#These are the core metrics which we will use for our model
real_ari = []
real_perplexity = []
real_burst = []
fake_perplexity = []
fake_ari = []
fake_burst = []
real = []
generated = []

#Iterate through the rows of our dataframe
for index, row in df.iloc[300:400].iterrows():
    print(f"{index}/{len(df)}")
    #Do not append to array if there is incomplete data
    try:
        #Analyze text samples
        fake_analysis = evaluate_for_ai(row["generated"])
        real_analysis = evaluate_for_ai(row["real"])

        #Extract Insights
        real.append(row["real"])
        generated.append(row["generated"])
        real_ari.append(real_analysis["average_readability"])
        real_perplexity.append(real_analysis["average_perplexity"])
        real_burst.append(real_analysis["overall_burstiness"])

        
        #Extract insights for fake data
        fake_ari.append(fake_analysis["average_readability"])
        fake_perplexity.append(fake_analysis["average_perplexity"])
        fake_burst.append(fake_analysis["overall_burstiness"])
    except Exception as e:
        print(e)
        print("error: omitting data")
        pass

# Create new DataFrame
results_df = pd.DataFrame({'real_ari': real_ari, 'real_perplexity': real_perplexity, 'real_burstiness': real_burst, 'fake_ari': fake_ari, 'fake_perplexity': fake_perplexity, 'fake_burstiness': fake_burst})
# Save the DataFrame to a CSV file
results_df.to_csv('results_1.csv', index=False)


