import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

st.title('Inflation and Unemployment Comparison')

#dataset
df = pd.read_csv("C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/inflation interest unemployment.csv")

#Country data   
country_data = {
    "US": df[df['country'] == 'United States'][['year','Inflation, consumer prices (annual %)','Unemployment, total (% of total labor force) (national estimate)']],
    "Algeria": df[df['country'] == 'Algeria'][['year','Inflation, consumer prices (annual %)','Unemployment, total (% of total labor force) (national estimate)']]
}

Country = st.sidebar.selectbox("Select Country", list(country_data.keys()))

# Get selected country's unemployment and inflation data
selected_country_data = country_data.get(Country, pd.DataFrame())
selected_country_data.rename(columns={'Inflation, consumer prices (annual %)': 'Inflation', 'Unemployment, total (% of total labor force) (national estimate)': 'Unemployment', 'year': 'Year'}, inplace=True)
selected_country_data.set_index('Year', inplace=True)

# Plotting
sns.set(rc={'figure.figsize': (15, 10)})
ax = sns.lineplot(data=selected_country_data, markers=True, dashes=False)
ax.set_title('Unemployment and Inflation trend over the years')
ax.set_xlabel('Year')

# Streamlit display
st.pyplot(plt)

# Calculate DTW
distance = dtw.distance(np.array(selected_country_data['Inflation']), np.array(selected_country_data['Unemployment']))
st.write("The DTW distance is: {}".format(distance))

# Plot DTW visualization
path = dtw.warping_path(np.array(selected_country_data['Inflation']), np.array(selected_country_data['Unemployment']))
fig, ax = dtwvis.plot_warping(np.array(selected_country_data['Inflation']), np.array(selected_country_data['Unemployment']), path)
plt.gca().set_xlabel('Inflation')
plt.gca().set_ylabel('Unemployment')
plt.title('DTW Warping Path')
st.pyplot(fig)
