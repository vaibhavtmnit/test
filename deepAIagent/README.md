

### The Big Picture (The Office Analogy)

Imagine you are running a research office.
1.  **The Manager (`graph.py`)**: Hands out assignments.
2.  **The Workers (`agents.py`)**: Do the actual reading and writing.
3.  **The Notebooks (`tools.py`)**: Where the workers write down what they find so they don't forget.
4.  **The Instructions (Prompts)**: This is what you are asking about.

---

### Part 1: Where do the Prompts go?

In this system, there are **two types of prompts**. You need to know where to find them in the code I gave you.

#### 1. The "Who Are You?" Prompt (The System Prompt)
**Where is it?** Inside `agents.py`.
**What does it do?** It tells the AI: *"You are a researcher. Read the document line by line. Write notes to a file."*

If you want to change **how** the AI behaves (e.g., make it stricter, make it summarize better), you edit this section in `agents.py`:

```python
# In agents.py, look for this variable:

system_instructions = f"""
You are a Deep Research Worker. Your ID is '{topic_id}'.
Your Goal: Research the topic "{topic}" within the provided document.

You have access to {total_chunks} document chunks (indices 0 to {total_chunks - 1}).

Follow this STRICT process:
... (Rest of instructions)
"""
```
*   **Action:** You usually **don't** need to touch this unless you want to change the rules of the game.

#### 2. The "What Do You Want?" Prompt (The Research Goals)
**Where is it?** Inside `main.py`.
**What does it do?** These are the specific questions you want the AI to answer about your document.

If you want to research something different, you edit this list in `main.py`:

```python
# In main.py, look for this list:

RESEARCH_GOALS = [
    "What were the primary objectives of the project?",  # <--- Prompt A
    "What were the technological spin-offs?"             # <--- Prompt B
]
```
*   **Action:** **This is where you type your actual questions.** If you run the code, the system will spawn 2 agents because there are 2 questions here.

---

### Part 2: How to Setup and Run

Follow these exact steps to get it running on your computer.

#### Step 1: Create the Folder Structure
Create a new folder on your computer named `deep_researcher`. Inside that folder, create these 5 empty files:

1.  `config.py`
2.  `tools.py`
3.  `agents.py`
4.  `graph.py`
5.  `main.py`
6.  `.env` (This is a hidden file for your passwords)

*Paste the code provided in the previous response into the corresponding files.*

#### Step 2: The API Keys (The .env file)
The robot needs a brain (Azure OpenAI). You need to put your credentials in the `.env` file. Open `.env` and paste this (replace with your actual keys):

```text
AZURE_OPENAI_API_KEY=your_actual_secret_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4
```
*Note: If you don't have these, ask your cloud administrator.*

#### Step 3: Install the Libraries
Open your terminal (Command Prompt or Terminal app). Navigate to your folder:

```bash
cd path/to/deep_researcher
```

Install the required Python packages:

```bash
pip install langchain-openai langgraph deepagents python-dotenv pyyaml
```

#### Step 4: Run the System
Now, press the "Go" button by running the main file:

```bash
python main.py
```

---

### Part 3: What happens when you press Run?

Here is the play-by-play of what you will see:

1.  **Orchestration Starts**: The "Manager" sees you have 2 goals in `main.py`.
2.  **Workers Spawn**: It creates 2 separate AI workers (Worker 0 and Worker 1).
3.  **File Creation (Look at your folder!)**:
    *   While the script is running, open your folder window.
    *   You will see new files appear magically: `topic_0_evidence.yaml` and `topic_1_evidence.yaml`.
    *   **This is the AI taking notes.** You can open these text files to see the AI extracting quotes *while it works*.
4.  **Final Report**:
    *   Once the workers finish reading the chunks, they read their own notes.
    *   They write a final answer.
    *   The terminal will print **=== FINAL AGGREGATED REPORT ===**.

### Summary Checklist
1.  **Code**: Copy/Paste the python files I gave you.
2.  **Keys**: Fill in `.env`.
3.  **Your Questions**: Edit `RESEARCH_GOALS` in `main.py`.
4.  **Run**: Type `python main.py`.
