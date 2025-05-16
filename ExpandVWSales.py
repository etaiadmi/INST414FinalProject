import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from pandastable import Table
import scipy.spatial.distance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm



country_data = pd.read_csv('world-data-2023.csv')
democracy_data = pd.read_csv('democracy-eiu.csv')
individualism_data = pd.read_csv('individualistic-countries-2025.csv')
vwsales = pd.read_csv('vwsales.csv')

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

df = df.merge(vwsales, left_on='Country', right_on='country', how='left')
print("Length merged data", len(df))


df = df[["Country", "IndividualismScore", "democracy_eiu", "Density\n(P/Km2)", 
                "Land Area(Km2)", "Urban_population", "CPI", "CPI Change (%)",
                "Life expectancy", "Tax revenue (%)", "sales2022"]]
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
                "Urban_population": "Urban Population", 
                "Unemployment rate": "Unemployment Rate (%)",
                "democracy_eiu": "Democracy Index", 
                "IndividualismScore": "Individualism Score",
                "sales2022": "Volume Sold (2022)",})

df.to_csv('CountriesData.csv', index=False)

#Only normalize for similarity
df_sim = df.copy()
df_reg = df.copy()

df_sim = df_sim.drop(columns=["Volume Sold (2022)"])

#Fill na with median
df_sim.iloc[:, 1:] = df_sim.iloc[:, 1:].apply(lambda x: x.fillna(x.median()))

#Normalization
df_sim.iloc[:, 1:] = MinMaxScaler().fit_transform(df_sim.iloc[:, 1:])


# #Set index to country
df_sim = df_sim.set_index('Country')

#Calculate similarity
query_countries = ["United States", "China", "Germany"]

for column in df_sim.columns:
    df_sim[column] = df_sim[column].astype(float)

for target_country in query_countries:
    target = df_sim.loc[target_country]
    distances = scipy.spatial.distance.cdist(df_sim,[target], metric='euclidean').flatten()
    query_distances = list(zip(df_sim.index, distances))
    most_similiar = []

    print(f"10 most similar countries to {target_country}:")
    for country, distance_score, in sorted(query_distances, key=lambda x: x[1])[1:11]:
        print(country, ":", distance_score)
        most_similiar.append({
            "Country": country, 
            "Distance": distance_score})
    pd.DataFrame(most_similiar).to_csv(f'MostSimilar{target_country}.csv', index=False)
    
    print()


#Run regression

#Fill na with median
cols_to_fill = [col for col in df_reg.columns if col != "Volume Sold (2022)" and col != "Country"]
df_reg[cols_to_fill] = df_reg[cols_to_fill].apply(lambda x: x.fillna(x.median()))


# Split into known and unknown sales
df_train = df_reg[df_reg['Volume Sold (2022)'].notna()]
df_predict = df_reg[df_reg['Volume Sold (2022)'].isna()]

print("Number of countries to predict:", len(df_predict))


# Select X features (exclude Country and Volume Sold 2022)
feature_cols = [col for col in df.columns if col not in ['Country', 'Volume Sold (2022)']]
X_train = df_train[feature_cols]
y_train = df_train['Volume Sold (2022)']
X_predict = df_predict[feature_cols]

# Train regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# After fitting the model
y_train_pred = reg.predict(X_train)

# Calculate metrics
r2 = r2_score(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("R² on training data:", r2)
print("Root Mean Squared Error on training data:", rmse)

# Fit the model using statsmodels
# Ensure X_train and y_train are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, invalid parsing will be NaN
y_train = pd.to_numeric(y_train, errors='coerce')  # Convert the target variable to numeric

X_train_sm = sm.add_constant(X_train)  # Adds the intercept to the model
model = sm.OLS(y_train, X_train_sm)  # Ordinary Least Squares regression
results = model.fit()

# Get the coefficients and p-values
coefficients = results.params  # Exclude the intercept
p_values = results.pvalues  # Exclude the intercept

# Create a DataFrame with feature names, coefficients, and p-values
coeff_df = pd.DataFrame({
    'Feature': ["Intercept"] + list(X_train.columns),
    'Coefficient': coefficients,
    'P-Value': p_values
})

# Export the coefficients DataFrame to a CSV file
coeff_df.to_csv('Model_Coefficients.csv', index=False)

# Predict sales for unknown countries
predicted_sales = reg.predict(X_predict)

# Create a new DataFrame with predictions
df_predicted_sales = df_predict[['Country']].copy()

df_predicted_sales['Predicted Volume Sold 2022'] = predicted_sales

# Optionally save the predicted dataset
df_predicted_sales.to_csv('CountriesData_with_PredictedSales.csv', index=False)




#Show df
root = tk.Tk()
root.title("Pandas DataFrame Viewer")

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

table = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
table.show()

root.mainloop()
