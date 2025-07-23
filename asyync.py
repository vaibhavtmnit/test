import asyncio
import time
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# --- Configuration ---
# It's recommended to use a .env file to store your API key securely.
# Create a file named .env in the same directory and add the following line:
# OPENAI_API_KEY="your-sk-..."
load_dotenv()

# Ensure the API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")

# --- Main Asynchronous Function ---

async def process_batch_concurrently():
    """
    This function demonstrates how to process a large batch of inputs
    asynchronously using a LangChain LLM instance.
    """
    print("Initializing LangChain components...")

    # 1. Define the processing chain using LangChain Expression Language (LCEL)
    # This is the modern and recommended way to build chains in LangChain.

    # The prompt template defines the input structure. Here, it expects a dictionary
    # with a "topic" key.
    prompt = ChatPromptTemplate.from_template(
        "Write a single, concise sentence that explains the core concept of {topic}."
    )

    # The model to use. gpt-4o-mini is a good choice for balancing cost, speed, and capability.
    # The model must support asynchronous calls.
    model = ChatOpenAI(model="gpt-4o-mini")

    # The output parser converts the LLM's complex output (a ChatMessage object)
    # into a simple string.
    output_parser = StrOutputParser()

    # We chain the components together using the pipe operator `|`.
    # The flow is: input -> prompt -> model -> output_parser -> final_string
    chain = prompt | model | output_parser

    print("Chain initialized successfully.")

    # 2. Generate the list of 40,000 inputs
    num_inputs = 40000
    print(f"Generating {num_inputs} sample inputs...")
    # These are placeholder topics. In a real application, this would be your actual data.
    topics = [f"the theory of relativity part {i+1}" for i in range(num_inputs)]
    # The `abatch` method expects a list of dictionaries, where each dictionary
    # corresponds to the input variables of the prompt.
    inputs_list = [{"topic": t} for t in topics]
    print(f"Generated {len(inputs_list)} inputs.")

    # 3. Process all inputs asynchronously using the .abatch() method
    print("\nStarting asynchronous processing of all inputs with chain.abatch()...")
    start_time = time.time()

    # `abatch` is the core of this process. It takes an iterable of inputs
    # and runs them through the chain concurrently. This is far more efficient
    # than a for-loop with `ainvoke`.
    # You can control the level of concurrency with the `max_concurrency` config key.
    # The default value is usually sufficient, but you can tune it.
    results = await chain.abatch(inputs_list, config={"max_concurrency": 100})

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {len(results)} inputs.")
    print(f"Total time taken: {total_time:.2f} seconds.")

    # 4. Display a few results for verification
    print("\n--- Sample Results ---")
    for i in range(min(5, len(results))):
        print(f"  Input Topic: '{topics[i]}'")
        print(f"  LLM Output: '{results[i]}'")
        print("-" * 20)

# --- Entry Point ---

if __name__ == "__main__":
    # To run an async function from a synchronous entry point, we use asyncio.run()
    try:
        asyncio.run(process_batch_concurrently())
    except Exception as e:
        print(f"An error occurred: {e}")




import asyncio
import time
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

# --- Configuration for Azure ---
# It's recommended to use a .env file to store your Azure OpenAI credentials.
# Create a file named .env in the same directory and add the following lines:
# AZURE_OPENAI_API_KEY="your-azure-api-key"
# AZURE_OPENAI_ENDPOINT="https://your-azure-openai-resource.openai.azure.com/"
# OPENAI_API_VERSION="2023-12-01-preview"
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="your-deployment-name"
load_dotenv()

# Ensure the necessary Azure environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]
if not all(os.getenv(var) for var in required_vars):
    raise ValueError(
        "One or more required Azure environment variables are not set. "
        "Please check your .env file or environment settings."
    )

# --- Main Asynchronous Function ---

async def process_batch_concurrently():
    """
    This function demonstrates how to process a large batch of inputs
    asynchronously using an AzureChatOpenAI instance with `ainvoke`.
    """
    print("Initializing LangChain components with AzureChatOpenAI...")

    # 1. Define the processing chain using LangChain Expression Language (LCEL)
    prompt = ChatPromptTemplate.from_template(
        "Write a single, concise sentence that explains the core concept of {topic}."
    )

    # The model is now AzureChatOpenAI. It automatically reads the required
    # credentials and endpoint details from the environment variables.
    model = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )

    output_parser = StrOutputParser()

    # The chain remains the same: input -> prompt -> model -> output_parser
    chain = prompt | model | output_parser

    print("Chain initialized successfully.")

    # 2. Generate the list of 40,000 inputs
    num_inputs = 40000
    print(f"Generating {num_inputs} sample inputs...")
    topics = [f"the theory of relativity part {i+1}" for i in range(num_inputs)]
    inputs_list = [{"topic": t} for t in topics]
    print(f"Generated {len(inputs_list)} inputs.")

    # 3. Process all inputs asynchronously using asyncio.gather with chain.ainvoke()
    print("\nStarting asynchronous processing with asyncio.gather and chain.ainvoke()...")
    start_time = time.time()

    # Create a list of tasks (coroutines), one for each input.
    # Each task is a call to chain.ainvoke().
    tasks = [chain.ainvoke(input_item) for input_item in inputs_list]

    # `asyncio.gather` runs all the tasks in the list concurrently.
    # While this works well, for very large numbers of inputs, the `abatch`
    # method is often more memory-efficient as it can stream results.
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {len(results)} inputs.")
    print(f"Total time taken: {total_time:.2f} seconds.")

    # 4. Display a few results for verification
    print("\n--- Sample Results ---")
    for i in range(min(5, len(results))):
        print(f"  Input Topic: '{topics[i]}'")
        print(f"  LLM Output: '{results[i]}'")
        print("-" * 20)

# --- Entry Point ---

if __name__ == "__main__":
    # To run an async function from a synchronous entry point, we use asyncio.run()
    try:
        asyncio.run(process_batch_concurrently())
    except Exception as e:
        print(f"An error occurred: {e}")



import asyncio
import time
import os
from dotenv import load_dotenv
from typing import List

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

# --- Configuration for Azure ---
# It's recommended to use a .env file to store your Azure OpenAI credentials.
# Create a file named .env in the same directory and add the following lines:
# AZURE_OPENAI_API_KEY="your-azure-api-key"
# AZURE_OPENAI_ENDPOINT="https://your-azure-openai-resource.openai.azure.com/"
# OPENAI_API_VERSION="2023-12-01-preview"
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="your-deployment-name"
load_dotenv()

# Ensure the necessary Azure environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]
if not all(os.getenv(var) for var in required_vars):
    raise ValueError(
        "One or more required Azure environment variables are not set. "
        "Please check your .env file or environment settings."
    )

# --- Pydantic Model for Structured Output ---
# Define the desired output structure. The model will validate the LLM's output.
class AnalysisResult(BaseModel):
    summary: str = Field(description="A concise summary of the topic.")
    relevance_score: int = Field(description="A relevance score from 1 to 10 based on the topic's importance.")

# --- Main Asynchronous Function ---

async def process_item(chain, input_item, semaphore):
    """
    A worker function to process a single item with semaphore control.
    This allows for retries on an individual item basis.
    """
    async with semaphore:
        try:
            # The chain is already configured with retries, so we just invoke it.
            result = await chain.ainvoke(input_item)
            return result
        except Exception as e:
            print(f"Request failed for input {input_item.get('topic', 'N/A')} after retries: {e}")
            # Return None or a specific error object for failed requests
            return None

async def process_batch_concurrently():
    """
    This function processes a large batch of inputs asynchronously with
    concurrency control, retries, and structured Pydantic output.
    """
    print("Initializing LangChain components with AzureChatOpenAI...")

    # 1. Define the processing chain using System/Human messages and Pydantic output
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert analyst. Analyze the user's topic and provide a summary and a relevance score."
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "Please analyze the following topic: {topic}"
    )
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # The model is now configured to automatically retry on failure and to
    # output structured data according to the Pydantic model.
    model = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    ).with_structured_output(AnalysisResult)

    # Add the .with_retry() method to the chain for automatic retries on transient errors.
    # It will retry 2 times after the initial failure (3 attempts total).
    chain = (prompt | model).with_retry(stop_after_attempt=3)

    print("Chain initialized with retries and structured output.")

    # 2. Generate the list of 40,000 inputs
    num_inputs = 40000
    print(f"Generating {num_inputs} sample inputs...")
    topics = [f"the theory of relativity part {i+1}" for i in range(num_inputs)]
    inputs_list = [{"topic": t} for t in topics]
    print(f"Generated {len(inputs_list)} inputs.")

    # 3. Process all inputs with concurrency control
    # Set the maximum number of concurrent requests.
    concurrency_limit = 50
    semaphore = asyncio.Semaphore(concurrency_limit)
    print(f"\nStarting asynchronous processing with a concurrency limit of {concurrency_limit}...")
    start_time = time.time()

    tasks = [process_item(chain, input_item, semaphore) for input_item in inputs_list]
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # Filter out None results from failed requests
    successful_results = [res for res in results if res is not None]

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed {len(successful_results)} out of {len(inputs_list)} inputs.")
    print(f"Total time taken: {total_time:.2f} seconds.")

    # 4. Display a few results for verification
    print("\n--- Sample Results (Validated Pydantic Objects) ---")
    for i in range(min(5, len(successful_results))):
        print(f"  Input Topic: '{topics[i]}'")
        # The result is now a Pydantic object, not just a string
        print(f"  LLM Output: {successful_results[i]}")
        print(f"  Summary: {successful_results[i].summary}")
        print(f"  Relevance Score: {successful_results[i].relevance_score}")
        print("-" * 20)

# --- Entry Point ---

if __name__ == "__main__":
    try:
        asyncio.run(process_batch_concurrently())
    except Exception as e:
        print(f"An error occurred: {e}")



import asyncio
import time
import os
import random
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

# --- Configuration for Azure ---
# It's recommended to use a .env file to store your Azure OpenAI credentials.
load_dotenv()

# Ensure the necessary Azure environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]
if not all(os.getenv(var) for var in required_vars):
    raise ValueError(
        "One or more required Azure environment variables are not set. "
        "Please check your .env file or environment settings."
    )

# --- Pydantic Model for Type-Controlled String Output ---
# This model ensures the final output of any chain is a validated string.
class StringOutput(BaseModel):
    response: str = Field(description="The final text output from the language model.")

# --- Asynchronous Worker and Main Function ---

async def process_task(task: Dict[str, Any], semaphore: asyncio.Semaphore):
    """
    A generic worker function to process a single task from any chain.
    It uses a semaphore to limit concurrency and appends the result to the correct list.
    """
    async with semaphore:
        try:
            # Get the chain, input, and result list from the task dictionary
            chain = task["chain"]
            input_data = task["input"]
            result_list = task["result_list"]
            
            # The chain is already configured with retries and structured output
            result = await chain.ainvoke(input_data)
            
            # Append the successful result (the Pydantic object) to the designated list
            if result:
                result_list.append(result)
            return True
        except Exception as e:
            print(f"Request failed for input {input_data.get('topic', 'N/A')} with chain {task.get('name', 'N/A')}: {e}")
            return False

async def run_chains_on_shared_inputs(agents: List[Dict[str, Any]], inputs: List[Dict[str, Any]], concurrency_limit: int):
    """
    This function takes a list of agents (chains) and a list of shared inputs,
    runs all inputs through all chains concurrently, and collects the results.
    """
    master_task_list = []
    
    # For each input, create a task for each agent
    for agent_config in agents:
        for input_data in inputs:
            task = {
                "chain": agent_config["chain"],
                "input": input_data,
                "result_list": agent_config["results"], # The list to append to
                "name": agent_config["name"]
            }
            master_task_list.append(task)
            
    # Shuffle the list to simulate requests coming in a random order
    random.shuffle(master_task_list)

    # --- Process all tasks with global concurrency control ---
    semaphore = asyncio.Semaphore(concurrency_limit)
    total_requests = len(master_task_list)
    
    print(f"\nStarting {total_requests} asynchronous requests with a global concurrency limit of {concurrency_limit}...")
    start_time = time.time()

    # Create and run all asyncio tasks
    asyncio_tasks = [process_task(task, semaphore) for task in master_task_list]
    await asyncio.gather(*asyncio_tasks)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Processing Complete ---")
    print(f"Total time taken: {total_time:.2f} seconds.")

    # --- Display Final Results from Each Agent's List ---
    print("\n--- Final Results ---")
    for agent_config in agents:
        num_requests = len(inputs)
        results = agent_config["results"]
        print(f"\n--- {agent_config['name']} Chain Results ({len(results)}/{num_requests} successful) ---")
        for result in results[:3]: # Print first 3
            print(f"  - {result.response}")


# --- Entry Point ---
if __name__ == "__main__":
    # --- Define Multiple LLM Chains ---
    print("Initializing multiple, distinct LangChain chains...")

    base_model = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )

    # Chain 1: A creative poet
    poet_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("You are a famous poet. Write a short, two-line poem about the user's topic."),
            HumanMessagePromptTemplate.from_template("{topic}")
        ]
    )
    poet_chain = (
        poet_prompt 
        | base_model.with_structured_output(StringOutput)
    ).with_retry(stop_after_attempt=3)

    # Chain 2: A factual scientist
    scientist_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("You are a research scientist. Explain the user's topic in one simple, factual sentence."),
            HumanMessagePromptTemplate.from_template("{topic}")
        ]
    )
    scientist_chain = (
        scientist_prompt
        | base_model.with_structured_output(StringOutput)
    ).with_retry(stop_after_attempt=3)

    print("Chains initialized successfully.")
    
    # --- Define Agents and Shared Inputs ---
    
    # 1. Define the list of agents to run. Each agent has a name, a chain,
    #    and a list where its results will be stored.
    agents_to_run = [
        {"name": "Poet", "chain": poet_chain, "results": []},
        {"name": "Scientist", "chain": scientist_chain, "results": []}
    ]
    
    # 2. Define the single list of inputs to be used by all agents
    shared_inputs = [{"topic": f"the concept of time {i+1}"} for i in range(10)]
    
    # 3. Set the total number of concurrent requests allowed across ALL agents
    global_concurrency_limit = 8

    try:
        # Pass the agents list, inputs list, and concurrency limit to the main function
        asyncio.run(run_chains_on_shared_inputs(
            agents=agents_to_run, 
            inputs=shared_inputs, 
            concurrency_limit=global_concurrency_limit
        ))
    except Exception as e:
        print(f"An error occurred: {e}")


import asyncio
import time
import os
import random
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

# --- Configuration for Azure ---
# It's recommended to use a .env file to store your Azure OpenAI credentials.
load_dotenv()

# Ensure the necessary Azure environment variables are available
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]
if not all(os.getenv(var) for var in required_vars):
    raise ValueError(
        "One or more required Azure environment variables are not set. "
        "Please check your .env file or environment settings."
    )

# --- Pydantic Model for Type-Controlled String Output ---
# This model ensures the final output of any chain is a validated string.
class StringOutput(BaseModel):
    response: str = Field(description="The final text output from the language model.")

# --- Asynchronous Worker and Main Function ---

async def process_task(task: Dict[str, Any], semaphore: asyncio.Semaphore):
    """
    A generic worker function to process a single task from any chain.
    It uses a semaphore to limit concurrency and appends the result to the correct list.
    """
    async with semaphore:
        try:
            # Get the chain, input, and result list from the task dictionary
            chain = task["chain"]
            input_data = task["input"]
            result_list = task["result_list"]
            
            # The chain is already configured with retries and structured output
            result = await chain.ainvoke(input_data)
            
            # Append the successful result (the Pydantic object) to the designated list
            if result:
                result_list.append(result)
            return True
        except Exception as e:
            print(f"Request failed for input {input_data.get('file_name', 'N/A')} with chain {task.get('name', 'N/A')}: {e}")
            return False

async def run_chains_on_shared_inputs(agents: List[Dict[str, Any]], inputs: List[Dict[str, Any]], concurrency_limit: int):
    """
    This function takes a list of agents (chains) and a list of shared inputs,
    runs all inputs through all chains concurrently, and collects the results.
    """
    master_task_list = []
    
    # For each input file, create a task for each agent
    for agent_config in agents:
        for input_data in inputs:
            task = {
                "chain": agent_config["chain"],
                "input": input_data,
                "result_list": agent_config["results"], # The list to append to
                "name": agent_config["name"]
            }
            master_task_list.append(task)
            
    # Shuffle the list to simulate a more realistic, unordered workload
    random.shuffle(master_task_list)

    # --- Process all tasks with global concurrency control ---
    semaphore = asyncio.Semaphore(concurrency_limit)
    total_requests = len(master_task_list)
    
    print(f"\nStarting {total_requests} asynchronous requests ({len(inputs)} files x {len(agents)} agents) with a global concurrency limit of {concurrency_limit}...")
    start_time = time.time()

    # Create and run all asyncio tasks
    asyncio_tasks = [process_task(task, semaphore) for task in master_task_list]
    await asyncio.gather(*asyncio_tasks)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Processing Complete ---")
    print(f"Total time taken: {total_time:.2f} seconds.")

    # --- Display Final Results from Each Agent's List ---
    print("\n--- Final Results Summary ---")
    for agent_config in agents:
        num_requests = len(inputs)
        results = agent_config["results"]
        print(f"\n--- {agent_config['name']} Chain Results ({len(results)}/{num_requests} successful) ---")
        # Print a few sample results for each agent
        for result in results[:2]: # Print first 2
            print(f"  - {result.response}")


# --- Entry Point ---
if __name__ == "__main__":
    # --- Define Multiple LLM Chains (Agents) ---
    print("Initializing 5 distinct agent chains...")

    base_model = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )
    
    # The human prompt now expects a 'file_content' variable
    human_message_prompt = HumanMessagePromptTemplate.from_template("{file_content}")

    # Agent 1: Summarizer
    summarizer_prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template("You are a world-class summarizer. Provide a concise, one-paragraph summary of the following text."), human_message_prompt])
    summarizer_chain = (summarizer_prompt | base_model.with_structured_output(StringOutput)).with_retry(stop_after_attempt=3)

    # Agent 2: Keyword Extractor
    keyword_prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template("You are an SEO expert. Extract the top 5 most relevant keywords from the following text, separated by commas."), human_message_prompt])
    keyword_chain = (keyword_prompt | base_model.with_structured_output(StringOutput)).with_retry(stop_after_attempt=3)

    # Agent 3: Sentiment Analyzer
    sentiment_prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template("You are a sentiment analysis bot. Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL."), human_message_prompt])
    sentiment_chain = (sentiment_prompt | base_model.with_structured_output(StringOutput)).with_retry(stop_after_attempt=3)

    # Agent 4: Question Generator
    question_prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template("You are a teacher. Based on the following text, generate one insightful question that would test a student's understanding."), human_message_prompt])
    question_chain = (question_prompt | base_model.with_structured_output(StringOutput)).with_retry(stop_after_attempt=3)

    # Agent 5: Analogy Creator
    analogy_prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template("You are a creative thinker. Create a simple analogy to explain the core concept of the following text."), human_message_prompt])
    analogy_chain = (analogy_prompt | base_model.with_structured_output(StringOutput)).with_retry(stop_after_attempt=3)

    print("All agent chains initialized successfully.")
    
    # --- Define Agents and Shared Inputs ---
    
    # 1. Define the list of agents to run.
    agents_to_run = [
        {"name": "Summarizer", "chain": summarizer_chain, "results": []},
        {"name": "Keyword Extractor", "chain": keyword_chain, "results": []},
        {"name": "Sentiment Analyzer", "chain": sentiment_chain, "results": []},
        {"name": "Question Generator", "chain": question_chain, "results": []},
        {"name": "Analogy Creator", "chain": analogy_chain, "results": []},
    ]
    
    # 2. Simulate reading 1000 files. In a real application, you would read
    #    from the filesystem here.
    print("\nSimulating the reading of 1000 files...")
    num_files = 1000
    shared_inputs = [
        {
            "file_name": f"document_{i}.txt", 
            "file_content": f"This is the content for file number {i}. The main topic is asynchronous programming and its benefits in modern applications, especially for I/O-bound tasks like API calls."
        } 
        for i in range(num_files)
    ]
    print(f"{len(shared_inputs)} file inputs prepared.")
    
    # 3. Set the total number of concurrent requests allowed across ALL agents
    global_concurrency_limit = 50

    try:
        # Pass the agents list, inputs list, and concurrency limit to the main function
        asyncio.run(run_chains_on_shared_inputs(
            agents=agents_to_run, 
            inputs=shared_inputs, 
            concurrency_limit=global_concurrency_limit
        ))
    except Exception as e:
        print(f"An error occurred: {e}")
