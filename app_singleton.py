import asyncio
from openai import OpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, FileSearchTool, Runner
import socketio
from quart import Quart, render_template, Response, request, redirect, session, stream_with_context, jsonify, url_for, flash, Response
from dotenv import load_dotenv
from agents import WebSearchTool, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.model_settings import ModelSettings
import pandas as pd
import os
import magic
from pdfminer.high_level import extract_text
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import json
import datetime
from werkzeug.utils import secure_filename
import requests
import logging
from pathlib import Path
import docx
# Using pypdf instead of PyPDF2
from pypdf import PdfReader

#load_dotenv()

# Set up logging
#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger(__name__)

# Initialize OpenAI client - uses API key from environment variable OPENAI_API_KEY
client = OpenAI()

# Create a vector store for document retrieval
# (OpenAI API call: https://platform.openai.com/docs/api-reference/vector-stores/create)
vector_store = client.vector_stores.create(name="GLPKDocs")

# Define data files to ingest
file_paths = ["glpk.pdf"]
# Open files in binary mode for upload
file_streams = [open(path, "rb") for path in file_paths]

# Batch upload files to vector store with automatic polling
# (OpenAI API call: https://platform.openai.com/docs/api-reference/vector-stores/file-batches/upload-and-poll)
file_batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams
)

# Print upload status summary
print("Upload status: ", file_batch.status)
print("Number of files uploaded: ", file_batch.file_counts.completed)

# Configure retrieval tool for the agent
tools = [
    FileSearchTool(
        vector_store_ids=[vector_store.id],  # Use our created vector store
        max_num_results=3,                  # Maximum documents to retrieve per query
        include_search_results=True         # Include raw content in agent context
    )
]

async def main():
    # Initialize custom agent instance
    agent = Agent(
        tools=tools,
        name="greenbldgagent",
        instructions="You are a glpk optimization expert.",  # System prompt
    )
    
    # Execute agent with streaming response
    result = Runner.run_streamed(
        agent,
        input="Explain how to solve the hvac problem for maintaining the building's office temperature at 24 degC but must keep energy efficiency at 32%. Show the glpk math expression."  # User query
    )
    
    # Stream and display response tokens in real-time
    async for event in result.stream_events():
        # Filter for text delta events (partial response chunks)
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)  # Stream to console

if __name__ == "__main__":
    asyncio.run(main())