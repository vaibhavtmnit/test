Sweet — I put a full working scaffold in the canvas with a clean LangGraph backend and a Chainlit front-end that does exactly what you described:

Supervisor → Query Parser → Lin Filter → Fetch+Accuracy → Chat loop

Proper gating to subject “ABC”

Looping back when fields are missing/invalid

lin_filter builds a pandas DataFrame row and pushes JSON for Chainlit to render

“Run accuracy” intent triggers a fetch+accuracy workflow (placeholder tools you can swap out)

After results, Supervisor flips to chat mode and answers only from AgentState context

Chainlit shows only user-facing messages + JSON payloads (no tool logs)

How to run

create venv & install:

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


start the UI:

chainlit run app/chainlit_app.py -w


(or try the CLI runner: python -m app.main)

Where to customize

Subject: change SUBJECT_NAME in app/graph/nodes/supervisor.py.

Required fields: edit REQUIRED_FIELDS in app/graph/nodes/query_parser.py.

LLM calls: replace model_invoke in app/graph/tools.py with your Azure OpenAI (or other) client.

Data access: implement data_source_tool(...) for your source.

Your accuracy module: call it inside run_accuracy_core(...) (currently a stub).

Chainlit integration notes

Nodes append UI-safe events to state["ui_events"]:

{"kind":"message","role":"assistant","text":"..."} → shows as a normal chat reply

{"kind":"dataframe","name":"lin_filter_df","json":[...]} → rendered as a JSON viewer

The Chainlit app streams LangGraph events and only renders these UI events, so tool outputs remain hidden.

To show additional DataFrames later, just append more dataframe events in any node.

If you want, I can swap in your real accuracy function and your actual source fetch next, or tailor the prompts/required fields to your ABC domain right away.
