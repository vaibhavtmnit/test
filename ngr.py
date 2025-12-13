async def run_manual_local_search(query: str):
    print(f"--- QUERY: {query} ---\n")

    # STEP 1: RETRIEVAL & CONTEXT BUILDING
    # -----------------------------------------------------
    # This single method performs the complex graph traversal:
    # 1. Embeds the query.
    # 2. Queries LanceDB for top entities.
    # 3. Pulls relationships connected to those entities.
    # 4. Pulls text chunks (sources) connected to those entities.
    
    print("STEP 1: Building Context (Vector Search + Graph Traversal)...")
    
    context_text, context_records = await search_engine.context_builder.build_context(
        query=query,
        conversation_history=None,
        include_entity_names=list(entities_df['name'].unique()), # Optional optimization
        
        # PARAMETERS controlling the window size
        max_tokens=12000,
        min_entity_rank=0,
        include_relationship_weight=True,
        top_k_mapped_entities=10, # How many entities to grab from vector store
        include_text_units=True,
    )
    
    # --- VISUALIZE THE "INGREDIENTS" ---
    
    print(f"\n[Stats] Context built with length: {len(token_encoder.encode(context_text))} tokens")
    
    # A. Entities Found
    print("\n--- A. TOP ENTITIES FOUND (From Vector Search) ---")
    if 'entities' in context_records:
        # Columns usually: [entity, description, rank, weight...]
        print(context_records['entities'][['entity', 'description']].head(5))
    
    # B. Relationships Found
    print("\n--- B. RELEVANT RELATIONSHIPS (Neighbors) ---")
    if 'relationships' in context_records:
        # Columns usually: [source, target, description, weight]
        print(context_records['relationships'][['source', 'target', 'description']].head(5))

    # C. Text Units (Raw Source Data)
    print("\n--- C. SUPPORTING TEXT UNITS (Source Chunks) ---")
    if 'sources' in context_records:
        print(context_records['sources'][['id', 'text']].head(2))


    # STEP 2: PROMPT CONSTRUCTION
    # -----------------------------------------------------
    print("\nSTEP 2: Constructing LLM Prompt...")
    
    # This is exactly how LocalSearch constructs the messages
    system_prompt = search_engine.system_prompt
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_text}\n\n### Query:\n{query}"}
    ]
    
    print(f"[System Prompt Snippet]: {system_prompt[:100]}...")
    print(f"[User Prompt Snippet (Context)]: {context_text[:200]}...")


    # STEP 3: GENERATION
    # -----------------------------------------------------
    print("\nSTEP 3: Generating Answer...")
    
    response = await search_engine.llm.agenerate(
        messages=messages,
        streaming=False,
        **search_engine.llm_params
    )
    
    print("\n--- FINAL LOCAL SEARCH RESPONSE ---")
    print(response.output)


if __name__ == "__main__":
    asyncio.run(run_manual_local_search("Who is the most influential person in this dataset?"))
