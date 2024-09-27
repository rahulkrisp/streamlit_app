import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

from streamlit import title
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textwrap3 import wrap

import os
from pathlib import Path
from dotenv import load_dotenv

import json
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials



load_dotenv()  # Load variables from .env file

# Get the Google credentials JSON from the environment variable
# if "STREAMLIT_SERVER" in os.environ:
    # Load the credentials from Streamlit secrets
    creds_toml = st.secrets["GOOGLE_CREDS"]
    creds_dict = json.loads(creds_toml)  # Parse if stored as a JSON string
# else:
#     # Load the credentials from .env file locally
#     creds_json_str = os.getenv("GOOGLE_CREDS")
#     # if not creds_json_str:
#     #     raise ValueError("GOOGLE_CREDS environment variable is not set or is empty.")
#     creds_dict = json.loads(creds_json_str)  # Parse the local JSON string

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

#
# headers = {
#     "authorization": st.secrets["GOOGLE_CREDS"],
# }




# Open the Google Sheet
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1Xy3zo-XMGVLgl7lVzR9gcDR5e9Sa-hw8LM-JhseTu8w/edit?pli=1&gid=704561408#gid=704561408').worksheet('Sheet5')

# Get the data into a pandas DataFrame
data = sheet.get_all_records()
df = pd.DataFrame(data)



# # Load  Excel file
# excel_file = 'C:/Users/user/Desktop/crispchat.xlsx'
# df = pd.read_excel(excel_file)

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Summary'] = df['Summary'].astype(str)
df['Summary'] = df['Summary'].apply(lambda x: '<br>'.join(wrap(x, width=30)))

# Sidebar date filter
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Filter data based on the selected date range
filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

# Drop rows where 'Plan' or 'Date' is NaN
# filtered_df = filtered_df.dropna(subset=['Plan'])
filtered_df = filtered_df[~filtered_df['Plan'].isin(['N/A', '', 'NA'])]  # Remove rows where Competitor Name is 'N/A' or 'NULL'
# Count occurrences of each plan type
plan_counts = filtered_df['Plan'].value_counts().reset_index()
plan_counts.columns = ['Plan Type', 'Count']

# Sort the plan counts in descending order
plan_counts = plan_counts.sort_values(by='Count', ascending=False)
#
# # Print plan_counts to verify sorting
# st.write(plan_counts)

# Create a horizontal bar plot
fig = px.bar(plan_counts,
             x='Count',
             y='Plan Type',
             orientation='h',
             title='Distribution of Plan Types',
             category_orders={'Plan Type': plan_counts['Plan Type']})

# Show the plot in the Streamlit app
st.plotly_chart(fig)




#WORDCLOUD

# Prepare the data
df['Content'] = df['Content'].fillna('')
all_content = ', '.join(df['Content'])

# Split by comma and strip spaces
words = [word.strip() for word in all_content.split(',')]

# Count occurrences of each word
word_counts = Counter(words)

# Generate word cloud
wordcloud = WordCloud(
    width=800, height=400,
    colormap='plasma',
    background_color='black',  # Word cloud background
    contour_color='white',  # Contour color for visibility
    contour_width=1
).generate_from_frequencies(word_counts)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Set the figure and axis backgrounds to transparent
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

# Display the word cloud
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

# Set title
plt.title('Word Cloud of Content', color='white', loc='left', pad=20, fontdict={'family': 'Arial', 'weight': 'bold'} )

# Save the figure with transparent background
st.pyplot(fig, transparent=True)







# Extract data inside parentheses
def clean_and_capitalize(text):
    # Remove unwanted phrases (case insensitive)
    text = re.sub(r'\(\s*country\s*,?\s*name\s*,?\s*:\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*country\s*,?\s*name\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*country\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*country:\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*name\s*\)', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\(\s*country\s*,?\s*name\s*,?\s*:\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*country\s*:\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*country\s*name\s*:\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*Country\s*name\s*:\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*country\s*\)', '', text, flags=re.IGNORECASE)

    # Capitalize the first letter of each word
    text = text.strip()  # Strip leading/trailing whitespace
    if text:
        text = text[0].upper() + text[1:]  # Capitalize the first character

    return text  # Return cleaned text



# Clean the Content column
df['Cleaned_Content'] = df['Content'].apply(clean_and_capitalize)

## Create a mapping for similar terms
mapping = {
    'Us': 'USA',
    'Usa': 'USA',
    'United States': 'USA',
    'U.s.': 'USA',
    'US': 'USA',
    'U.K.': 'UK',
    'U.K': 'UK',
    'United Kingdom': 'UK',
    'Uae': 'UAE',
    'U.a.e.': 'UAE',
    'United Arab Emirates': 'UAE',
    'Canada': 'Canada',
    'Australia': 'Australia',
    'AUS': 'Australia',
    'Brasil': 'Brazil',
    'Brazil': 'Brazil',
    'Germany': 'Germany',
    'Italy': 'Italy',
}

# Replace similar terms with a single form
df['Cleaned_Content'] = df['Cleaned_Content'].replace(mapping, regex=True)

# Extract data inside parentheses (if any)
df['Country'] = df['Cleaned_Content'].str.extract(r'\((.*?)\)')[0]

# Count occurrences of each unique value
country_counts = df['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

# Display the results in Streamlit
import streamlit as st

st.markdown("<h5 style='font-size: 16px;'><strong>Country Count Table</strong></h5>", unsafe_allow_html=True)
st.dataframe(country_counts)




#PIECHART

df['User Sentiment'] = df['User Sentiment'].astype(str)  # Ensure all values are strings
df = df.assign(User_Sentiment=df['User Sentiment'].str.split(',')).explode('User_Sentiment')

# Trim any extra spaces
df['User_Sentiment'] = df['User_Sentiment'].str.strip()

# Count unique sentiments
sentiment_counts = df['User_Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

top_sentiments = sentiment_counts.head(8)

# Create a doughnut chart
fig_doughnut = px.pie(top_sentiments,
             values='Count',
             names='Sentiment',
             title='User Sentiment Distribution',
             hole=0.4)  # This makes it a doughnut chart

# Show the plot in the Streamlit app
st.plotly_chart(fig_doughnut)





# SENTIMENT SCATTER PLOT

def extract_sentiment(text):
    try:
        # Remove any trailing characters like '}'
        text = text.replace('}', '').replace('{', '')
        # Split the text by commas
        parts = text.split(',')
        # Extract values
        polarity = float(parts[0].split(':')[1].strip())
        subjectivity = float(parts[1].split(':')[1].strip())
        return pd.Series([polarity, subjectivity], index=['polarity', 'subjectivity'])
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
        return pd.Series([None, None], index=['polarity', 'subjectivity'])
# Apply the extraction function
df[['polarity', 'subjectivity']] = df['Python User Sentiment'].apply(extract_sentiment)
fig_scatter = px.scatter(df, x='polarity', y='subjectivity', title='Sentiment Analysis: Polarity vs. Subjectivity', hover_data={'Summary': True})

fig_scatter.update_traces(
    hovertemplate="<br>".join([
        "Polarity: %{x:.2f}<br>",
        "Subjectivity: %{y:.2f}<br>",
        "<b>Summary:</b> %{customdata[0]}<br>",

    ])
)

# Display in Streamlit
st.plotly_chart(fig_scatter)

print(df.dtypes)





# SENTIMENT TABLE

st.markdown("<h5 style='font-size: 16px;'><strong>Sentiment Analysis</h5>", unsafe_allow_html=True)

# Function to clean the summary
def clean_summary(text):
    return text.replace('<br>', ' ')

# Clean the summaries
df['Summary'] = df['Summary'].apply(clean_summary)  # This line applies the clean_summary function

# Manual input for polarity and subjectivity range
polarity_min = st.number_input("Enter Minimum Polarity", -1.0, 1.0, value=-1.0)
polarity_max = st.number_input("Enter Maximum Polarity", -1.0, 1.0, value=1.0)

# Filter DataFrame based on the selected ranges
filtered_df = df[
    (df['polarity'] >= polarity_min) & (df['polarity'] <= polarity_max)]
# Sort by date, newest first
filtered_df = filtered_df.sort_values(by='Date', ascending=False)

# Pagination
page_size = 10
total_pages = (len(filtered_df) // page_size) + (1 if len(filtered_df) % page_size > 0 else 0)
page = st.number_input("Select Page", min_value=1, max_value=total_pages, value=1)

# Get the current page data
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size
current_page_df = filtered_df.iloc[start_idx:end_idx]

# If there are fewer than 10 records, show the top 20 values based on the latest date
if len(current_page_df) == 0:
    st.write("No records found in the selected range.")
else:
    # Create a table with relevant data
    st.markdown("<h5 style='font-size: 16px;'><strong>Sentiment Analysis Results</h5>", unsafe_allow_html=True)
    current_page_df['SessionID'] = current_page_df['SessionID'].apply(lambda x: f"https://app.crisp.chat/website/{x}")
    current_page_df = current_page_df[['Date', 'Email', 'polarity', 'subjectivity', 'Summary', 'SessionID']].reset_index(drop=True)
    st.dataframe(current_page_df)

# Show pagination information
st.write(f"Page {page} of {total_pages}")

# If the data is less than 10 rows and pagination is active, show top values
if len(filtered_df) < page_size:
    top_df = filtered_df.head(20)
    st.write("### Top Sentiment Analysis Results")
    top_df['SessionID'] = top_df['SessionID'].apply(lambda x: f"[{x}](http://example.com/{x})")
    top_df = top_df[['Date', 'Email', 'polarity', 'subjectivity', 'Summary', 'SessionID']]
    st.dataframe(top_df)





# COMPETITOR PLOT
import plotly.graph_objects as go


# Handle potential NaN values in 'Competitor Details' by replacing NaN with an empty string
df['Competitor Details'] = df['Competitor Details'].fillna('')

# Filter out rows where 'Competitor Name' is NaN, 'N/A', or 'NULL'
df = df[~df['Competitor Name'].isin(['N/A', '', 'NA'])]  # Remove rows where Competitor Name is 'N/A' or 'NULL'
df = df.dropna(subset=['Competitor Name'])  # Remove rows where Competitor Name is NaN

# Count the occurrences of each competitor
competitor_counts = df['Competitor Name'].value_counts().reset_index()
competitor_counts.columns = ['Competitor Name', 'Count']

# Group by 'Competitor Name' and concatenate 'Competitor Details'
competitor_details = df.groupby('Competitor Name')['Competitor Details'].apply(lambda x: '<br>'.join(x)).reset_index()

# Merge counts with details
summary_df = pd.merge(competitor_counts, competitor_details, on='Competitor Name')

# Get top 15 competitors by count
top_15_summary_df = summary_df.nlargest(15, 'Count')

# Format competitor details as a bulleted list for hover info
top_15_summary_df['Competitor Details'] = top_15_summary_df['Competitor Details'].apply(
    lambda x: '<br>'.join([f'• {detail}' for detail in x.split('<br>')])
)

# Create a Plotly bar chart with hovertemplate
fig_competitor = go.Figure(data=[go.Bar(
    x=top_15_summary_df['Competitor Name'],
    y=top_15_summary_df['Count'],
    text=top_15_summary_df['Count'],
    texttemplate='%{text}',
    hovertemplate=(
        '<b>%{x}</b><br>' +
        'Count: %{y}<br>' +
        'Details:<br>%{customdata}'
    ),
    customdata=top_15_summary_df['Competitor Details']
)])

# Update layout to improve readability
fig_competitor.update_layout(
    xaxis_title='Competitor Name',
    yaxis_title='Number of Mentions',
    xaxis_tickangle=-45,  # Tilt x-axis labels
    xaxis_title_standoff=25,
    title='Top 15 Competitors by Mentions',
    title_x=0, #Left aligh the title
    title_font_size=18
)

# Display the Plotly chart
st.plotly_chart(fig_competitor)

# # Add your visualization components here
# if not filtered_df.empty:
#     # Example: Line chart for numerical data
#     numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns
#     if numeric_columns:
#         for col in numeric_columns:
#             st.subheader(f"{col} over Time")
#             fig = px.line(filtered_df, x='date', y=col)
#             st.plotly_chart(fig)
#
#     # Example: Bar chart for categorical data
#     categorical_columns = filtered_df.select_dtypes(include=['object']).columns
#     if categorical_columns:
#         for col in categorical_columns:
#             st.subheader(f"{col} Distribution")
#             fig = px.bar(filtered_df, x=col, color=col)
#             st.plotly_chart(fig)
# else:
#     st.warning("No data available for the selected date.")