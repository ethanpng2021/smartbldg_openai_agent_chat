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

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

class ConfigClass(object):
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'docx', 'doc', 'pdf'}
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'Uploads')

app = Quart(__name__)
app.config.from_object(__name__ + '.ConfigClass')

asgi_app = socketio.ASGIApp(sio, app)

#vector_store = client.vector_stores.create(name="glpk")

# Global list for vector store IDs
vector_store_ids = []

# Initial files
initial_file_paths = ["glpk.pdf"]
if initial_file_paths:
    for path in initial_file_paths:
        if os.path.exists(path):
            vs = client.vector_stores.create(name=os.path.basename(path))
            with open(path, "rb") as f:
                file_batch = client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vs.id,
                    files=[f]
                )
            if file_batch.status == 'completed':
                vector_store_ids.append(vs.id)
                print(f"Initial file {path} uploaded to vector store {vs.id}")

# Function to get updated tools
def get_tools():
    return [
        FileSearchTool(
            vector_store_ids=vector_store_ids,
            max_num_results=3,
            include_search_results=True
        )
    ]

# Initialize the agent
def get_agent():
    return Agent(
        tools=get_tools(),
        name="greenbldgagent",
        instructions="You are an expert in green building controls and HVAC systems. Use the document retrieval tool when needed to answer questions based on uploaded documents.",
    )

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
async def index():
    return await render_template("index.html")

@app.route('/llm', methods=['POST'])
async def llm():
    global vector_store_ids
    try:
        form = await request.form
        files = form.getlist('file')
        message = form.get('message', '').strip()

        uploaded_messages = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                content = await file.read()

                # Create new vector store for this file
                vs = client.vector_stores.create(name=f"Doc_{filename}")

                file_tuple = (filename, content, file.mimetype)
                vs_file = client.vector_stores.files.create(
                    vector_store_id=vs.id,
                    file=file_tuple
                )

                # Poll until the file is processed
                while vs_file.status == 'in_progress':
                    await asyncio.sleep(1)
                    vs_file = client.vector_stores.files.retrieve(
                        vector_store_id=vs.id,
                        file_id=vs_file.id
                    )

                if vs_file.status == 'failed':
                    raise Exception(f"File processing failed for {filename}")

                vector_store_ids.append(vs.id)
                uploaded_messages.append(f"File '{filename}' uploaded and processed in vector store {vs.id}.\n")

        # Get updated agent with new tools
        agent = get_agent()

        async def generate():
            if uploaded_messages:
                yield ''.join(uploaded_messages)

            if message:
                result = Runner.run_streamed(agent, input=message)
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        yield event.data.delta #+ '\n'
            elif not uploaded_messages:
                yield "No message or files provided."

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error in /llm: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)