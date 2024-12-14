# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "ipykernel",
#   "openai",
#   "numpy",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from pathlib import Path
import asyncio

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Ensure token is retrieved from environment variable
def get_token():
    try:
        return os.environ["AIPROXY_TOKEN"]
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

async def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding)

async def async_post_request(headers, data):
    """Async function to make HTTP requests."""
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

async def generate_narrative(analysis, token, file_path):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # Prepare the prompt for narrative generation
    prompt = (
        f"You are a data analyst. Provide a detailed narrative based on the following data analysis results for the file '{file_path.name}':\n\n"
        f"Column Names & Types: {list(analysis['summary'].keys())}\n\n"
        f"Summary Statistics: {analysis['summary']}\n\n"
        f"Missing Values: {analysis['missing_values']}\n\n"
        f"Correlation Matrix: {analysis['correlation']}\n\n"
        "Based on this information, please provide insights into any trends, outliers, anomalies, "
        "or patterns you can detect. Suggest additional analyses that could provide more insights, such as clustering, anomaly detection, etc."
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    return await async_post_request(headers, data)

async def analyze_data(df, token):
    """Use LLM to suggest and perform data analysis."""
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)

    # Prepare the prompt to ask the LLM for analysis suggestions
    prompt = (
        f"You are a data analyst. Given the following dataset information, provide an analysis plan:\n\n"
        f"Columns: {list(df.columns)}\n"
        f"Data Types: {df.dtypes.to_dict()}\n"
        f"First 5 rows of data:\n{df.head()}\n\n"
        "Please suggest useful data analysis techniques, such as correlation analysis, regression, anomaly detection, clustering, or others."
    )
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    # Requesting analysis suggestions from the LLM
    suggestions = await async_post_request(headers, data)
    print(f"LLM Suggestions: {suggestions}")

    # Basic analysis (summary statistics, missing values, correlations)
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {}
    }
    print("Data analysis complete.")
    return analysis, suggestions

async def visualize_data(df, output_dir, analysis):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit to 3 distribution plots for numeric columns
    for idx, column in enumerate(numeric_columns[:3]):
        plt.figure(figsize=(6, 6))
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        file_name = output_dir / f'{column}_distribution.png'
        plt.savefig(file_name, dpi=100)
        print(f"Saved distribution plot: {file_name}")
        plt.close()

    # Generate one correlation heatmap (if numeric columns exist)
    if numeric_columns.any():
        plt.figure(figsize=(6, 6))
        corr = df[numeric_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        file_name = output_dir / 'correlation_heatmap.png'
        plt.savefig(file_name, dpi=100)
        print(f"Saved correlation heatmap: {file_name}")
        plt.close()

async def save_narrative_with_images(narrative, output_dir):
    """Save narrative to README.md and embed image links."""
    readme_path = output_dir / 'README.md'
    image_links = "\n".join(
        [f"![{img.name}]({img.name})" for img in output_dir.glob('*.png')]
    )
    with open(readme_path, 'w') as f:
        f.write(narrative + "\n\n" + image_links)
    print(f"Narrative successfully written to {readme_path}")

async def main(file_path):
    print("Starting autolysis process...")

    # Ensure input file exists
    file_path = Path(file_path)
    if not file_path.is_file():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    # Load token
    token = get_token()

    # Load dataset
    df = await load_data(file_path)
    print("Dataset loaded successfully.")

    # Analyze data with LLM insights
    print("Analyzing data...")
    analysis, suggestions = await analyze_data(df, token)
    print(f"LLM Analysis Suggestions: {suggestions}")

    # Create output directory
    output_dir = Path(file_path.stem)  # Create a directory named after the dataset
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations with LLM suggestions
    print("Generating visualizations...")
    await visualize_data(df, output_dir, analysis)

    # Generate narrative
    print("Generating narrative using LLM...")
    narrative = await generate_narrative(analysis, token, file_path)

    if narrative != "Narrative generation failed due to an error.":
        await save_narrative_with_images(narrative, output_dir)
    else:
        print("Narrative generation failed. Skipping README creation.")

    print("Autolysis process completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)
    
    # Run the main function in an event loop
    asyncio.run(main(sys.argv[1]))
