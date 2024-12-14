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
#   "scipy",
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
import scipy.stats as stats

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Ensure token is retrieved from environment variable
def get_token():
    try:
        return os.environ["AIPROXY_TOKEN"]
    except KeyError as e:
        print(f"Error: Environment variable '{e.args[0]}' not set.")
        raise

async def load_data(file_path):
    """Load CSV data with encoding detection."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(file_path, encoding=encoding)

async def async_post_request(headers, data):
    """Async function to make HTTP requests."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            print(f"Error during request: {e}")
            raise

async def generate_narrative(analysis, token, file_path):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    prompt = (
        f"You are a data analyst. Provide a detailed narrative based on the following data analysis results for the file '{file_path.name}':\n\n"
        f"Column Names & Types: {list(analysis['summary'].keys())}\n\n"
        f"Summary Statistics: {analysis['summary']}\n\n"
        f"Missing Values: {analysis['missing_values']}\n\n"
        f"Correlation Matrix: {analysis['correlation']}\n\n"
        "Please provide insights into trends, outliers, anomalies, or patterns. "
        "Suggest further analyses like clustering or anomaly detection. "
        "Discuss how these trends may impact future decisions."
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    return await async_post_request(headers, data)

async def analyze_data(df, token):
    """Use LLM to suggest and perform data analysis."""
    if df.empty:
        raise ValueError("Error: Dataset is empty.")

    # Enhanced prompt for better LLM analysis suggestions
    prompt = (
        f"You are a data analyst. Given the following dataset information, provide an analysis plan and suggest useful techniques:\n\n"
        f"Columns: {list(df.columns)}\n"
        f"Data Types: {df.dtypes.to_dict()}\n"
        f"First 5 rows of data:\n{df.head()}\n\n"
        "Suggest data analysis techniques, such as correlation, regression, anomaly detection, clustering, or others. "
        "Consider missing values, categorical variables, and scalability."
    )

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        suggestions = await async_post_request(headers, data)
    except Exception as e:
        suggestions = f"Error fetching suggestions: {e}"

    print(f"LLM Suggestions: {suggestions}")

    # Basic analysis (summary statistics, missing values, correlations)
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {}
    }

    # Hypothesis testing example (if 'A' and 'B' columns exist)
    if 'A' in df.columns and 'B' in df.columns:
        t_stat, p_value = stats.ttest_ind(df['A'].dropna(), df['B'].dropna())
        analysis['hypothesis_test'] = {
            't_stat': t_stat,
            'p_value': p_value
        }

    print("Data analysis complete.")
    return analysis, suggestions

async def visualize_data(df, output_dir):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Select main columns for distribution based on importance
    selected_columns = numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enhanced visualizations (distribution plots, heatmap)
    for column in selected_columns:
        plt.figure(figsize=(6, 6))
        sns.histplot(df[column].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution of {column}')
        file_name = output_dir / f'{column}_distribution.png'
        plt.savefig(file_name, dpi=100)
        print(f"Saved distribution plot: {file_name}")
        plt.close()

    if len(numeric_columns) > 1:
        plt.figure(figsize=(8, 8))
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
    try:
        token = get_token()
    except Exception as e:
        print(e)
        sys.exit(1)

    # Load dataset
    try:
        df = await load_data(file_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    print("Dataset loaded successfully.")

    # Analyze data with LLM insights
    print("Analyzing data...")
    try:
        analysis, suggestions = await analyze_data(df, token)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"LLM Analysis Suggestions: {suggestions}")

    # Create output directory
    output_dir = Path(file_path.stem)  # Create a directory named after the dataset
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations with LLM suggestions
    print("Generating visualizations...")
    await visualize_data(df, output_dir)

    # Generate narrative
    print("Generating narrative using LLM...")
    narrative = await generate_narrative(analysis, token, file_path)

    if narrative != "Narrative generation failed due to an error.":
        await save_narrative_with_images(narrative, output_dir)
    else:
        print("Narrative generation failed.")

# Execute script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
