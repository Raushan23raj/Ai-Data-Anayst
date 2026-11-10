import os
import io
import json
import uuid
import hashlib
import secrets
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.request import Request, urlopen

from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from dotenv import load_dotenv

app = Flask(__name__)
# The session cookie only holds a random id; the history itself lives on the server.
# Set FLASK_SECRET_KEY in .env to keep browsers "logged in" across server restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing from the .env file.")

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------
MAX_STORED_TURNS = 50          # turns kept per session
MAX_PROMPT_TURNS = 5           # turns sent to the LLM as context
MAX_SESSIONS = 200             # oldest sessions are dropped beyond this
RESULT_PREVIEW_CHARS = 1500    # how much of each raw result is sent back to the LLM

# sid -> {"file_hash": str | None, "file_name": str | None, "turns": [ ... ]}
SESSIONS = OrderedDict()
SESSIONS_LOCK = threading.Lock()


def get_chat():
    """Return the chat state for this browser session, creating it if needed."""
    sid = session.get("sid")
    with SESSIONS_LOCK:
        if sid not in SESSIONS:
            sid = uuid.uuid4().hex
            session["sid"] = sid
            SESSIONS[sid] = {"file_hash": None, "file_name": None, "turns": []}
        SESSIONS.move_to_end(sid)
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
        return SESSIONS[sid]


def preview_result(raw_result):
    """Short text version of a result, safe to put back into a prompt."""
    if isinstance(raw_result, pd.DataFrame):
        text = raw_result.to_string(max_rows=20, max_cols=15)
    elif isinstance(raw_result, pd.Series):
        text = raw_result.to_string(max_rows=20)
    else:
        text = str(raw_result)
    if len(text) > RESULT_PREVIEW_CHARS:
        text = text[:RESULT_PREVIEW_CHARS] + "\n... (truncated)"
    return text


def format_history_for_prompt(turns, include_code=True):
    """Render the most recent turns as text for the LLM."""
    recent = turns[-MAX_PROMPT_TURNS:]
    if not recent:
        return ""
    blocks = []
    for number, turn in enumerate(recent, start=1):
        block = f"Q{number}: {turn['question']}\n"
        if include_code:
            block += f"Code used:\n{turn['code']}\nRaw result:\n{turn['result_preview']}\n"
        block += f"Answer given: {turn['answer']}"
        blocks.append(block)
    return "\n\n".join(blocks)


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

    if not file or not question:
        return jsonify({"error": "Please provide a CSV file and a question."}), 400

    # 2. Load the CSV into a Pandas DataFrame
    try:
        file_bytes = file.read()
        df = pd.read_csv(io.BytesIO(file_bytes))
        columns = df.columns.tolist()
        dataset_profile = build_dataset_profile(df)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"})

    # 2b. Load this session's history. A different CSV means a different
    # conversation, so the old history is dropped.
    chat = get_chat()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    history_reset = False
    if chat["file_hash"] != file_hash:
        history_reset = bool(chat["turns"])
        chat["turns"] = []
        chat["file_hash"] = file_hash
        chat["file_name"] = file.filename
    history_text = format_history_for_prompt(chat["turns"])
    history_section = (
        f"\n    Previous conversation about this same dataset (oldest first):\n{history_text}\n"
        if history_text else ""
    )

    # 3. Step One: Generate Python/Pandas Code
    # We pass the column names (and previous turns) so the model knows the context.
    code_prompt = f"""
    You are a data analyst. I have a Pandas DataFrame named 'df'.
    The columns are: {columns}.
    The dataset profile is:
    {json.dumps(dataset_profile, indent=2)}
    {history_section}
    The user is now asking: "{question}"

    Write a short Python code block to answer this question.

    CRITICAL RULES:
    1. Assign the final answer to a variable named `result`.
    2. If computing correlation on text/categorical columns (e.g. 'Beginner', 'Intermediate', 'Advanced'), first convert those columns to numeric category codes using `.astype('category').cat.codes`.
    3. Use the supplied dtypes and categorical value counts to understand the data before writing the code.
    4. Keep the code simple and readable.
    5. Return ONLY the python code inside a ```python ``` code block.
    6. If the question refers to earlier turns (e.g. "it", "those", "the previous result", "now do the same for..."), use the previous conversation to work out what is meant. Variables from earlier code do NOT exist: `df` is always the original, unmodified data, so recompute anything you need (you may reuse the earlier code).

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
    qa_text = format_history_for_prompt(chat["turns"], include_code=False)
    qa_section = f"\n    Conversation so far (for context only):\n{qa_text}\n" if qa_text else ""
    result_preview = preview_result(raw_result)
    nl_prompt = f"""
    {qa_section}
    The user asked: "{question}"
    The raw data result from the database is: {result_preview}
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

    # 7. Save this turn to the session history
    chat["turns"].append({
        "question": question,
        "code": pandas_code,
        "answer": friendly_answer,
        "raw_result": raw_html,
        "result_preview": result_preview,
    })
    del chat["turns"][:-MAX_STORED_TURNS]

    # 8. Return the code, final answer, and raw output to the frontend
    return jsonify({
        "code": pandas_code,
        "answer": friendly_answer,
        "raw_result": raw_html,
        "history_reset": history_reset,
    })


@app.route('/history')
def history():
    """Return this session's saved conversation so the page can restore it."""
    chat = get_chat()
    return jsonify({
        "file_name": chat["file_name"],
        "turns": [
            {k: turn[k] for k in ("question", "answer", "code", "raw_result")}
            for turn in chat["turns"]
        ],
    })


@app.route('/clear', methods=['POST'])
def clear():
    """Start a fresh conversation."""
    chat = get_chat()
    chat["turns"] = []
    chat["file_hash"] = None
    chat["file_name"] = None
    return jsonify({"ok": True})


if __name__ == '__main__':
    # Run the server
    app.run(debug=True, port=5000)