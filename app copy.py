import os
import json
from pathlib import Path
from urllib.request import Request, urlopen

from flask import Flask, request, jsonify, render_template
import pandas as pd
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing from the .env file.")

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def build_dataset_profile(df):
    """Return dtypes and low-cardinality values from categorical columns."""
    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
    categorical_profile = {}

    for column in df.columns:
        series = df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue
        
        unique_values = series.unique()
        unique_count = int(series.nunique(dropna=False))
        column_profile = {"unique_value_count": unique_count}
        if unique_count < 20:
            column_profile["unique_values"] = [
                None if pd.isna(value) else str(value)
                for value in unique_values
            ]
        categorical_profile[column] = column_profile

    return {
        "dtypes": dtypes,
        "categorical_columns": categorical_profile,
    }


def ask_openrouter(prompt):
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"OpenRouter request failed: {error}") from error

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected OpenRouter response: {result}") from error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Get inputs from the frontend
    file = request.files.get('file')
    question = request.form.get('question')

    # 2. Load the CSV into a Pandas DataFrame
    try:
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        dataset_profile = build_dataset_profile(df)
        print(dataset_profile)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"})

    # 3. Step One: Generate Python/Pandas Code
    # We pass the column names so Gemini knows what the data looks like.
    code_prompt = f"""
    You are a data analyst. I have a Pandas DataFrame named 'df'.
    The columns are: {columns}.
    The dataset profile is:
    {json.dumps(dataset_profile, indent=2)}
    The user is asking: "{question}"
    
    Write a short Python code block to answer this question.
    
    CRITICAL RULES:
    1. Assign the final answer to a variable named `result`.
    2. If computing correlation on text/categorical columns (e.g. 'Beginner', 'Intermediate', 'Advanced'), first convert those columns to numeric category codes using `.astype('category').cat.codes`.
    3. Use the supplied dtypes and categorical value counts to understand the data before writing the code.
    4. Keep the code simple and readable.
    5. Return ONLY the python code inside a ```python ``` code block.
    
    Example:
    ```python
    # Calculate correlation after converting categorical columns
    df['Prompt_Engineering_Skill_encoded'] = df['Prompt_Engineering_Skill'].astype('category').cat.codes
    result = df['Prompt_Engineering_Skill_encoded'].corr(df['Post_Semester_GPA'])
    ```
    """
    try:
        raw_text = ask_openrouter(code_prompt)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 502

    # Helper parser to extract python code block
    if "```python" in raw_text:
        pandas_code = raw_text.split("```python")[1].split("```")[0].strip()
    elif "```" in raw_text:
        pandas_code = raw_text.split("```")[1].split("```")[0].strip()
    else:
        pandas_code = raw_text
    
    # 4. Step Two: Execute the Code using exec
    try:
        local_vars = {"df": df, "pd": pd}
        exec(pandas_code, {}, local_vars)
        raw_result = local_vars.get("result", "No result assigned to variable 'result'.")
    except Exception as e:
        return jsonify({"error": f"Code execution failed. Code:\n{pandas_code}\n\nError: {str(e)}"})

    # 5. Step Three: Convert to Natural Language
    nl_prompt = f"""
    The user asked: "{question}"
    The raw data result from the database is: {raw_result}
    Write a single, friendly, natural language sentence to deliver this answer to the user.
    """
    
    try:
        friendly_answer = ask_openrouter(nl_prompt)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 502

    # 6. Format the raw result for rendering
    if isinstance(raw_result, pd.DataFrame):
        raw_html = raw_result.to_html(classes="raw-table", border=0)
    elif isinstance(raw_result, pd.Series):
        raw_html = raw_result.to_frame().to_html(classes="raw-table", border=0)
    else:
        raw_html = f"<pre>{str(raw_result)}</pre>"

    # 7. Return the code, final answer, and raw output to the frontend
    return jsonify({
        "code": pandas_code,
        "answer": friendly_answer,
        "raw_result": raw_html
    })

if __name__ == '__main__':
    # Run the server
    app.run(debug=True, port=5000)