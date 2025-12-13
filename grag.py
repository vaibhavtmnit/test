
async def run_manual_global_search(query: str):
    print(f"--- QUERY: {query} ---\n")

    # STEP 1: Build Context (Prepare Batches)
    # -----------------------------------------------------
    # This prepares the data chunks (batches of community reports)
    context_result = await context_builder.build_context(
        query=query,
        conversation_history=None,
        **search_engine.context_builder_params,
    )
    
    print(f"STEP 1: Context Built. Created {len(context_result.context_chunks)} batches of reports.\n")

    # STEP 2: Map Step (Generate Intermediate Responses)
    # -----------------------------------------------------
    # We iterate over the chunks and call the internal map method manually.
    # In the real code, this is done via asyncio.gather for parallelism.
    
    map_responses = []
    
    print("STEP 2: Running Map Step (Intermediate Responses)...")
    for i, batch in enumerate(context_result.context_chunks):
        # Call the internal method _map_response_single_batch
        # This sends the 'map_system_prompt' + data to the LLM
        response = await search_engine._map_response_single_batch(
            context_data=batch,
            query=query,
            **search_engine.map_llm_params
        )
        map_responses.append(response)
        
        # --- SHOW OUTPUT OF MAP STEP ---
        print(f"\n[Batch {i+1} Map Output]:")
        # The response usually contains a list of 'points' with 'score'
        print(response.response) 
        # You can also inspect response.context_data to see which reports were used

    # STEP 3: Intermediate Aggregation & Ranking
    # -----------------------------------------------------
    # The _reduce_response method inside GlobalSearch handles filtering internally,
    # but we can simulate/inspect it here.
    
    print(f"\nSTEP 3: Aggregating & Ranking {len(map_responses)} Map Responses...")
    
    # Filter out responses that have no useful information (usually score=0 or empty)
    # Note: The structure of 'response' depends on your specific version/prompt,
    # but typically it's an object containing 'points'.
    
    # We can inspect the raw points collected
    all_points = []
    for r in map_responses:
        if r.response:
            # This logic depends on whether parsed_response is available or just raw text
            # Standard GraphRAG often returns a structured object or string.
            print(f" - Response Score: {getattr(r, 'score', 'N/A')}")
            all_points.append(r.response)

    # STEP 4: Reduce Step (Final Generation)
    # -----------------------------------------------------
    # We feed the collected map_responses into the reduce method.
    # This will filter the best points and generate the final answer.
    
    print("\nSTEP 4: Running Reduce Step (Final Answer)...")
    
    reduce_response = await search_engine._reduce_response(
        map_responses=map_responses,
        query=query,
        **search_engine.reduce_llm_params,
    )
    
    print("\n--- FINAL GLOBAL SEARCH RESPONSE ---")
    print(reduce_response.response)
    
    print("\n--- INTERMEDIATE DATA USED IN REDUCE ---")
    # This shows exactly which points from Step 2 made it into the final context
    print(reduce_response.context_text)

# Run the async loop
if __name__ == "__main__":
    asyncio.run(run_manual_global_search("What are the major themes in this dataset?"))
