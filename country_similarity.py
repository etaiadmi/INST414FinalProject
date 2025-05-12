import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from pandastable import Table
import scipy.spatial.distance


country_data = pd.read_csv('world-data-2023.csv')
democracy_data = pd.read_csv('democracy-eiu.csv')
individualism_data = pd.read_csv('individualistic-countries-2025.csv')

#Clean democracy data
democracy_data = democracy_data[democracy_data['Year'] == 2022]
continents = ['Africa', 'Asia', 'Europe', 
              'North America', 'Oceania', 'South America']
democracy_data = democracy_data[~democracy_data['Entity'].isin(continents)]
democracy_data = democracy_data.drop(columns=['Year', "Code"])
print("Length democracy data", len(democracy_data))

#Clean Individualism data
individualism_data = individualism_data.drop(columns=[
    'IndividualisticCountries_IndividualismScore_2023', "flagCode"])
print("Length individualism data", len(individualism_data))

#Clean country data
columns_keep = ["Country", 
                "Density\n(P/Km2)", 
                "Land Area(Km2)", 
                "CPI",
                "CPI Change (%)", 
                "Life expectancy",
                "Tax revenue (%)", 
                "Urban_population", 
                ]
country_data = country_data[columns_keep]
print("Length country data", len(country_data))

#Merge data
df = country_data.merge(democracy_data, left_on='Country', right_on='Entity', how='inner')
df = df.merge(individualism_data, left_on='Country', right_on='country', how='inner')
df = df.drop(columns=['Entity', 'country'])

df = df[["Country", "IndividualismScore", "democracy_eiu", "Density\n(P/Km2)", 
                "Land Area(Km2)", "Urban_population", "CPI", "CPI Change (%)",
                "Life expectancy", "Tax revenue (%)",]]
#Removes commas from data
df = df.apply(lambda x: x.str.replace(',', '') if x.dtype == "object" else x)

#Turns strings into floats
df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: float(x.replace('%', '')
                                                         .replace('$', '')
                                                         .replace(',', ''))
                                         if isinstance(x, str) else x)

#Save pd to csv, make columns pretty
df = df.rename(columns={"Density\n(P/Km2)": "Density (P/Km2)",
                "Land Area(Km2)": "Land Area (Km2)", 
                "Life expectancy": "Life Expectancy",
                "Urban_population": "Urban Population (%)", 
                "Unemployment rate": "Unemployment Rate (%)",
                "democracy_eiu": "Democracy Index", 
                "IndividualismScore": "Individualism Score"})

df.to_csv('Module3Countries.csv', index=False)

print("Length merged data", len(df))

#Fill na with median
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.fillna(x.median()))

#Normalization
df.iloc[:, 1:] = MinMaxScaler().fit_transform(df.iloc[:, 1:])

#Set index to country
df = df.set_index('Country')

#Calculate similarity
query_countries = ["United States", "China", "Canada"]

for column in df.columns:
    df[column] = df[column].astype(float)

for target_country in query_countries:
    target = df.loc[target_country]
    distances = scipy.spatial.distance.cdist(df,[target], metric='euclidean').flatten()
    query_distances = list(zip(df.index, distances))
    most_similiar = []

    print(f"10 most similar countries to {target_country}:")
    for country, distance_score, in sorted(query_distances, key=lambda x: x[1])[1:11]:
        print(country, ":", distance_score)
        most_similiar.append({
            "Country": country, 
            "Distance": distance_score})
    pd.DataFrame(most_similiar).to_csv(f'MostSimilar{target_country}.csv', index=False)
    
    print()


#Show df

root = tk.Tk()
root.title("Pandas DataFrame Viewer")

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

table = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
table.show()

root.mainloop()