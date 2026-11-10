# CSV Data Analyst Agent

An interactive Flask web application that allows users to upload CSV files and ask analytical questions in plain English. The application uses Google's Gemini API to automatically generate and execute Python/Pandas code on the data, and then translates the result back into a friendly natural language response.

## Features

- **Interactive CSV Upload**: Upload any CSV file.
- **Natural Language Queries**: Ask questions about your dataset in plain English.
- **AI-Powered Code Generation**: Uses Gemini to write targeted Pandas code.
- **Dynamic Execution**: Safely runs generated code against your dataset.
- **Friendly Explanations**: Synthesizes structured data results back into clean, readable answers.

---

## Prerequisites

Before running the application, make sure you have:
- Python 3.8 or higher installed.
- A Gemini API Key.

---

## Getting a Gemini API Key

To get a free or pay-as-you-go Gemini API key:

1. **Go to Google AI Studio**: Visit [Google AI Studio](https://aistudio.google.com/).
2. **Sign In**: Sign in using your Google Account.
3. **Generate Key**: 
   - Click the **Get API Key** button in the top left or top right corner.
   - Click **Create API Key**.
   - Choose to create the key in a new Google Cloud project or an existing one.
4. **Copy the Key**: Copy the generated API key (it starts with `AIzaSy...`).

---

## Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/demo_w
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     .venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install flask pandas google-generativeai
   ```

5. **Set the Gemini API Key environment variable:**
   - **macOS/Linux:**
     ```bash
     export GEMINI_API_KEY="your_api_key_here"
     ```
   - **Windows (Command Prompt):**
     ```cmd
     set GEMINI_API_KEY="your_api_key_here"
     ```
   - **Windows (PowerShell):**
     ```powershell
     $env:GEMINI_API_KEY="your_api_key_here"
     ```

---

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Upload a CSV file (e.g., `ai_student_impact_dataset_.csv` or `sales.csv`), enter a question, and click **Analyze**!
