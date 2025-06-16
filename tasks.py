import subprocess
import json
import sqlite3
import markdown
import datetime
from pathlib import Path
from typing import List
import asyncio
import base64
import os
from bs4 import BeautifulSoup
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import requests
# import speech_recognition as sr

PROXY_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
PROXY_MODEL = "gpt-4o-mini"
PROXY_API_KEY = os.environ["AIPROXY_TOKEN"]

DATE_FORMATS = [
    "%Y-%m-%d",  # 2024-03-14
    "%d-%b-%Y",  # 14-Mar-2024
    "%b %d, %Y",  # Mar 14, 2024
    "%Y/%m/%d",  # 2024/03/14
]


# Phase B: Ensure file access is within data
def restrict_file_access(path: Path) -> None:
    if str(path.resolve()).startswith(str(get_path("data"))):
        return
    else:
        raise Exception(f"Access to files outside data is forbidden: {path}")


def get_path(path: str) -> Path:
    project_dir = Path(
        __file__).resolve().parent  # Get the current project directory
    print(f"Project directory: {project_dir}")

    # Remove the leading '/' if it exists and join the path correctly
    if path.startswith('/'):
        path = path[1:]  # Remove the leading slash
    full_path = project_dir / Path(
        path)  # Join project_dir with the relative path

    print(f"Resolved file path: {full_path}")
    return full_path


# Use ThreadPoolExecutor to run blocking file IO tasks asynchronously
async def read_file(path: str) -> str:
    """
    Reads the content of the specified file asynchronously using ThreadPoolExecutor.
    Ensures the file is within the allowed data directory.
    """
    file_path = get_path(path)
    restrict_file_access(file_path)

    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, _read_file_sync, file_path)

    return content


# Synchronous function to read the file (to be run in a separate thread)
def _read_file_sync(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        return f.read()


# Prevent accidental file deletion
def prevent_file_deletion(path: Path) -> None:
    if path.exists():
        if path.stat().st_size == 0:
            raise Exception(f"Attempting to delete or truncate a file: {path}")
    if path.exists() and path.is_dir():
        raise Exception(
            f"Attempting to delete or truncate a directory: {path}")


async def execute_task(task_description: str) -> str:
    """
    Uses GPT-4 to determine which internal task function to call based on a natural language task description.
    If no suitable internal function is found, use the LLM to directly generate a response.
    """
    task_map = {
        "run_datagen": run_datagen,
        "run_format_markdown": run_format_markdown,
        "run_count_wednesdays": run_count_wednesdays,
        "run_sort_contacts": run_sort_contacts,
        "run_extract_log": run_extract_log,
        "run_generate_index": run_generate_index,
        "run_extract_email_sender": run_extract_email_sender,
        "run_extract_credit_card": run_extract_credit_card,
        "run_find_similar_comments": run_find_similar_comments,
        "run_calculate_gold_sales": run_calculate_gold_sales,
        "run_fetch_data_from_api": run_fetch_data_from_api,
        "run_clone_git_repo_and_commit": run_clone_git_repo_and_commit,
        "run_sql_query_on_db": run_sql_query_on_db,
        "run_scrape_website": run_scrape_website,
        "run_compress_or_resize_image": run_compress_or_resize_image,
        # "run_transcribe_audio": run_transcribe_audio,
        "run_convert_markdown_to_html": run_convert_markdown_to_html
    }

    task_list = "\n".join(f"- {k.replace('run_', '').replace('_', ' ')} → {k}"
                          for k in task_map.keys())
    prompt = f"""You're an assistant. Based on the user's task description, choose the most appropriate internal method name to execute from the list below. Only return the method name.
                Available functions:
                {task_list}
                Task: "{task_description}"
                Respond with just the function name like `run_extract_credit_card`."""

    response = requests.post(PROXY_API_URL,
                             headers={
                                 "Authorization": f"Bearer {PROXY_API_KEY}",
                                 "Content-Type": "application/json"
                             },
                             json={
                                 "model": PROXY_MODEL,
                                 "messages": [{
                                     "role": "user",
                                     "content": prompt
                                 }]
                             })
    response.raise_for_status()
    function_name = response.json()['choices'][0]['message']['content'].strip()

    if function_name not in task_map:
        # Fallback: No matching function, so just generate a direct reply with the LLM
        fallback_prompt = f"""You are an assistant. The user asked:
        "{task_description}"

        Since this doesn't match any internal task, please provide a helpful response directly."""
        fallback_response = requests.post(PROXY_API_URL,
                                          headers={
                                              "Authorization":
                                              f"Bearer {PROXY_API_KEY}",
                                              "Content-Type":
                                              "application/json"
                                          },
                                          json={
                                              "model":
                                              PROXY_MODEL,
                                              "messages": [{
                                                  "role":
                                                  "user",
                                                  "content":
                                                  fallback_prompt
                                              }]
                                          })
        fallback_response.raise_for_status()
        return fallback_response.json(
        )['choices'][0]['message']['content'].strip()

    func_params = {
        "run_datagen": ["user_email"],
        "run_format_markdown": ["file_name"],
        "run_count_wednesdays": ["file_name"],
        "run_sort_contacts": ["file_name"],
        "run_extract_log": [],
        "run_generate_index": [],
        "run_extract_email_sender": [],
        "run_extract_credit_card": ["file_name"],
        "run_find_similar_comments": ["file_name"],
        "run_calculate_gold_sales": ["file_name"],
        "run_fetch_data_from_api": ["api_url", "output_file"],
        "run_clone_git_repo_and_commit":
        ["user_email", "repo_url", "commit_message"],
        "run_sql_query_on_db": ["db_file", "sql_query", "output_file"],
        "run_scrape_website": ["url", "output_file"],
        "run_compress_or_resize_image":
        ["input_image", "output_image", "max_size"],
        # "run_transcribe_audio": ["input_audio", "output_file"],
        "run_convert_markdown_to_html": ["input_md_file", "output_html_file"],
    }

    param_list = func_params.get(function_name, [])
    if not param_list:
        return await task_map[function_name]()

    param_prompt = f"""You are given a natural language task description and a method name with its parameters.
                    Task description:
                    "{task_description}"

                    Method name: {function_name}
                    Parameters: {', '.join(param_list)}

                    Extract the value for each parameter from the task description. Return a JSON object where keys are parameter names and values are the extracted values as strings.
                    If a parameter is not mentioned in the task description, return null for its value.
                    Output only the JSON object."""

    param_response = requests.post(PROXY_API_URL,
                                   headers={
                                       "Authorization":
                                       f"Bearer {PROXY_API_KEY}",
                                       "Content-Type": "application/json"
                                   },
                                   json={
                                       "model":
                                       PROXY_MODEL,
                                       "messages": [{
                                           "role": "user",
                                           "content": param_prompt
                                       }]
                                   })
    #print("Raw LLM Response:", param_response.text)
    param_response.raise_for_status()
    param_json_str = param_response.json(
    )['choices'][0]['message']['content'].strip()

    try:
        cleaned_content = param_response.json(
        )['choices'][0]['message']['content'].strip()
        cleaned_content = cleaned_content.replace("```json",
                                                  "").replace("```",
                                                              "").strip()

        params = json.loads(cleaned_content)
        print(f"Extracted parameters: {params}")
    except Exception as e:
        print(f"Failed to parse parameters JSON: {e}")
        print("Response was:", param_response.text)
        raise

    missing_params = [p for p in param_list if not params.get(p)]
    if missing_params:
        # Parameters missing → fallback response
        fallback_prompt = f"""You are an assistant. The user asked:
        "{task_description}"

        The system could not extract required parameters {missing_params} for function {function_name}.
        Please provide a helpful response directly instead."""
        fallback_response = requests.post(PROXY_API_URL,
                                          headers={
                                              "Authorization":
                                              f"Bearer {PROXY_API_KEY}",
                                              "Content-Type":
                                              "application/json"
                                          },
                                          json={
                                              "model":
                                              PROXY_MODEL,
                                              "messages": [{
                                                  "role":
                                                  "user",
                                                  "content":
                                                  fallback_prompt
                                              }]
                                          })
        fallback_response.raise_for_status()
        return fallback_response.json(
        )['choices'][0]['message']['content'].strip()

    args = [params.get(p) for p in param_list]

    return await task_map[function_name](*args)


# Define Phase A Task functions
async def run_datagen(user_email: str) -> str:
    command = ["python3", "datagen.py", user_email, "--root", "data"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Failed to generate data")
    return "Data generation complete"


async def run_format_markdown(file_name: str) -> str:
    file_path = get_path("data/" + file_name)
    restrict_file_access(file_path)

    command = ["npx", "prettier", "--write", str(file_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Failed to format markdown")
    return "Markdown formatting complete"


async def run_count_wednesdays(file_name: str) -> str:
    """
    Count Wednesdays from the list of dates in data/datefile.txt or data/datetext.txt,
    supporting multiple date formats.
    """
    file_path = get_path(file_name)

    if not file_path:
        raise ValueError(
            "No valid input file found among 'datefile.txt' or 'datetext.txt'")

    with open(file_path, "r") as f:
        lines = f.readlines()

    wednesdays = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parsed_date = None
        for fmt in DATE_FORMATS:
            try:
                parsed_date = datetime.datetime.strptime(line, fmt)
                break
            except ValueError:
                continue
        if parsed_date and parsed_date.weekday() == 2:  # Wednesday is 2
            wednesdays += 1

    output_path = get_path("data/dates-wednesdays.txt")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write(str(wednesdays))

    return f"Counted {wednesdays} Wednesdays"


async def run_sort_contacts(file_name: str) -> str:
    file_path = get_path(file_name)
    restrict_file_access(file_path)

    with open(file_path, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts,
                             key=lambda x: (x['last_name'], x['first_name']))

    output_path = get_path("data/contacts-sorted.json")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        json.dump(sorted_contacts, f)
    return "Contacts sorted"


async def run_extract_log(user_email: str) -> str:
    logs = sorted(get_path("data/logs/").glob("*.log"),
                  key=lambda x: x.stat().st_mtime,
                  reverse=True)
    if not logs:
        raise Exception("No log files found")

    file_path = logs[0]
    restrict_file_access(file_path)

    with open(file_path, "r") as f:
        first_line = f.readline()

    output_path = get_path("data/logs-recent.txt")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write(first_line)
    return "Extracted first line of most recent log"


async def run_generate_index(user_email: str) -> str:
    index = {}
    for md_file in get_path("data/docs/").glob("*.md"):
        restrict_file_access(md_file)

        with open(md_file, "r") as f:
            for line in f:
                if line.startswith("# "):
                    index[md_file.name] = line.strip("# ").strip()
                    break

    output_path = get_path("data/docs/index.json")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        json.dump(index, f)
    return "Generated index of markdown files"


async def run_extract_email_sender(file_name: str) -> str:
    file_path = get_path(file_name)
    restrict_file_access(file_path)

    with open(file_path, "r") as f:
        email_content = f.read()

    sender = await extract_sender_from_email(email_content)

    output_path = get_path("data/email-sender.txt")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write(sender)
    return f"Extracted sender email: {sender}"


async def run_extract_credit_card(file_name: str) -> str:
    """
    A8: Extract credit card number from the image in data/credit-card.png
    """
    file_path = get_path(file_name)
    restrict_file_access(file_path)

    with open(file_path, "rb") as f:
        image_data = f.read()

    card_number = await extract_card_number_from_image(image_data)

    output_path = get_path("data/credit-card.txt")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write(card_number)
    return f"Extracted card number: {card_number}"


async def run_find_similar_comments(file_name: str) -> str:
    """
    A9: Find similar comments from data/comments.txt
    """
    file_path = get_path("data/" + file_name)
    restrict_file_access(file_path)

    with open(file_path, "r") as f:
        comments = f.readlines()

    most_similar = await find_most_similar_comments(comments)

    output_path = get_path("data/comments-similar.txt")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write("\n".join(most_similar))
    return "Found most similar comments"


async def run_calculate_gold_sales(file_name: str) -> str:
    """
    A10: Calculate gold sales from the SQLite database data/ticket-sales.db
    """
    db_path = get_path("data/" + file_name)
    restrict_file_access(db_path)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0]

    output_path = get_path("data/ticket-sales-gold.txt")
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write(str(total_sales))
    return f"Total sales for Gold tickets: {total_sales}"


async def extract_sender_from_email(email_content: str) -> str:
    prompt = f"Extract the sender's email address from this message:\n\n{email_content}"
    response = requests.post(PROXY_API_URL,
                             headers={
                                 "Authorization": f"Bearer {PROXY_API_KEY}",
                                 "Content-Type": "application/json"
                             },
                             json={
                                 "model": PROXY_MODEL,
                                 "messages": [{
                                     "role": "user",
                                     "content": prompt
                                 }]
                             })
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


async def find_most_similar_comments(comments: List[str]) -> List[str]:
    prompt = "Find the most similar pair of comments from the following list:\n\n" + "\n".join(
        comments)
    response = requests.post(PROXY_API_URL,
                             headers={
                                 "Authorization": f"Bearer {PROXY_API_KEY}",
                                 "Content-Type": "application/json"
                             },
                             json={
                                 "model": PROXY_MODEL,
                                 "messages": [{
                                     "role": "user",
                                     "content": prompt
                                 }]
                             })
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip().split(
        "\n")


async def extract_card_number_from_image(image_data: bytes) -> str:
    base64_image = base64.b64encode(image_data).decode("utf-8")
    image_url = {"url": f"data:image/png;base64,{base64_image}"}
    prompt = "Extract the credit card number from the following image data."

    response = requests.post(PROXY_API_URL,
                             headers={
                                 "Authorization": f"Bearer {PROXY_API_KEY}",
                                 "Content-Type": "application/json"
                             },
                             json={
                                 "model":
                                 "gpt-4-vision-preview",
                                 "messages": [{
                                     "role":
                                     "user",
                                     "content": [{
                                         "type": "text",
                                         "text": prompt
                                     }, {
                                         "type": "image_url",
                                         "image_url": image_url
                                     }]
                                 }]
                             })
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


async def run_fetch_data_from_api(api_url: str, output_file: str) -> str:
    """
    Fetch data from an API and save it to a file.
    """
    response = requests.get(api_url)
    response.raise_for_status()  # Ensure we get a successful response

    data = response.json()  # Assuming the API returns JSON data

    output_path = get_path(output_file)
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        json.dump(data, f)

    return f"Fetched data from API and saved to {output_file}"


async def run_clone_git_repo_and_commit(user_email: str, repo_url: str,
                                        commit_message: str) -> str:
    """
    Clone a Git repository and make a commit.
    """
    repo_dir = get_path(f"data/{user_email}_repo")
    prevent_file_deletion(repo_dir)

    # Clone the repository
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    # Navigate to the repo directory and make a commit
    subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(
        ["git", "-C",
         str(repo_dir), "commit", "-m", commit_message],
        check=True)
    subprocess.run(["git", "-C", str(repo_dir), "push"], check=True)

    return f"Cloned repo {repo_url} and made a commit with message: '{commit_message}'"


async def run_sql_query_on_db(db_file: str, sql_query: str,
                              output_file: str) -> str:
    """
    Run a SQL query on an SQLite or DuckDB database and save the result.
    """
    db_path = get_path(db_file)
    prevent_file_deletion(db_path)

    conn = sqlite3.connect(str(db_path))  # Use DuckDB if necessary
    cursor = conn.cursor()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    conn.close()

    # Save the results to a file
    output_path = get_path(output_file)
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        json.dump(results, f)

    return f"Executed query on {db_file} and saved results to {output_file}"


async def run_scrape_website(url: str, output_file: str) -> str:
    """
    Extract data from a website (scrape) and save it.
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure we get a successful response

    soup = BeautifulSoup(response.text, 'html.parser')
    extracted_data = []  # Parse the required data from the soup

    for element in soup.find_all("h1"):  # Example: extracting all h1 tags
        extracted_data.append(element.get_text())

    output_path = get_path(output_file)
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        json.dump(extracted_data, f)

    return f"Scraped data from {url} and saved to {output_file}"


async def run_compress_or_resize_image(input_image: str, output_image: str,
                                       max_size: tuple) -> str:
    """
    Compress or resize an image and save the result.
    """
    input_path = get_path(input_image)
    prevent_file_deletion(input_path)

    output_path = get_path(output_image)
    prevent_file_deletion(output_path)

    # Open image, resize, and save it
    with Image.open(input_path) as img:
        img.thumbnail(max_size)
        img.save(output_path)

    return f"Compressed and resized image from {input_image} to {output_image}"


# async def run_transcribe_audio(input_audio: str, output_file: str) -> str:
#     """
#     Transcribe audio from an MP3 file and save the transcript.
#     """
#     input_path = get_path(input_audio)
#     prevent_file_deletion(input_path)

#     recognizer = sr.Recognizer()

#     # Load the MP3 file into the recognizer
#     with sr.AudioFile(str(input_path)) as audio_file:
#         audio_data = recognizer.record(audio_file)

#     # Perform transcription
#     transcript = recognizer.recognize_google(audio_data)

#     output_path = get_path(output_file)
#     prevent_file_deletion(output_path)

#     with open(output_path, "w") as f:
#         f.write(transcript)

#     return f"Transcribed audio from {input_audio} and saved to {output_file}"


async def run_convert_markdown_to_html(input_md_file: str,
                                       output_html_file: str) -> str:
    """
    Convert a Markdown file to HTML and save the result.
    """
    input_path = get_path(input_md_file)
    prevent_file_deletion(input_path)

    with open(input_path, "r") as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content)

    output_path = get_path(output_html_file)
    prevent_file_deletion(output_path)

    with open(output_path, "w") as f:
        f.write(html_content)

    return f"Converted Markdown file {input_md_file} to HTML and saved to {output_html_file}"
