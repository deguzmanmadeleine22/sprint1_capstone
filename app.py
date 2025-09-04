import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# st.title("Hello Streamlit!")
# st.write("This is my first Streamlit app 🎉")
#
# number = st.slider("Pick a number", 0, 100, 50)
# st.write(f"You picked: {number}")

tips = sns.load_dataset("tips")

st.title("Seaborn Graph in Streamlit")

# Create a matplotlib figure
fig, ax = plt.subplots()
sns.barplot(data=tips, x="day", y="total_bill", estimator=sum, ax=ax)
ax.set_title("Total Bill per Day")

# Display the figure in Streamlit
st.pyplot(fig)