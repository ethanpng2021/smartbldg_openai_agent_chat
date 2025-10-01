import asyncio
import socketio
from quart import Quart, render_template, Response, request, redirect, session, stream_with_context, jsonify, url_for, flash, Response
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from agents import Agent, WebSearchTool, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def get_extension(filename):
    try: 
        extension = os.path.splitext(filename)[1]
        bitstr = extension.lstrip('.')
    except:
        bitstr = "Unsuccessful extraction"
    return bitstr

class WordFileProcessor:
    def __init__(self):
        self.supported_mime_types = {
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'application/msword': self._process_legacy_doc,
            'text/csv': self._process_csv,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._process_excel,
            'application/vnd.ms-excel': self._process_excel
        }
        self.extension_to_mime = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel'
        }

    def process_file_type(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        mime_type = magic.from_file(file_path, mime=True)
        file_extension = Path(file_path).suffix.lower()
        
        logger.debug(f"Processing file: {file_path}, Extension: {file_extension}, MIME: {mime_type}")
        
        # Handle application/octet-stream by checking file extension
        if mime_type == 'application/octet-stream' or mime_type is None:
            if file_extension in self.extension_to_mime:
                mime_type = self.extension_to_mime[file_extension]
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        
        if mime_type not in self.supported_mime_types:
            raise ValueError(f"Unsupported file type: {mime_type}")
        
        return self.supported_mime_types[mime_type](file_path)

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                metadata = pdf_reader.metadata or {}
                text = extract_text(file_path)  # Using pdfminer.high_level.extract_text
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
        
        return self._extract_word_data(text, metadata)

    def _process_docx(self, file_path: str) -> Dict[str, Any]:
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            metadata = {
                'author': doc.core_properties.author or "Unknown",
                'created': str(doc.core_properties.created) if doc.core_properties.created else "Unknown",
                'modified': str(doc.core_properties.modified) if doc.core_properties.modified else "Unknown"
            }
            return self._extract_word_data(text, metadata)
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            raise ValueError(f"Failed to process DOCX: {str(e)}")

    def _process_legacy_doc(self, file_path: str) -> Dict[str, Any]:
        raise ValueError("Legacy .doc files are not supported in this implementation. Please convert to .docx or .pdf.")

    def _process_csv(self, file_path: str) -> Dict[str, Any]:
        try:
            data = pd.read_csv(file_path)
            text = data.to_string()
            metadata = {
                'row_count': len(data),
                'columns': list(data.columns)
            }
            return self._extract_word_data(text, metadata)
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
            raise ValueError(f"Failed to process CSV: {str(e)}")

    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        try:
            data = pd.read_excel(file_path)
            text = data.to_string()
            metadata = {
                'row_count': len(data),
                'columns': list(data.columns)
            }
            return self._extract_word_data(text, metadata)
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {str(e)}")
            raise ValueError(f"Failed to process Excel: {str(e)}")

    def _extract_word_data(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        result = {
            'metadata': metadata,
            'extracted_text': text,
            'processing_date': datetime.datetime.now().isoformat()
        }
        
        return result

    def save_to_json(self, data: Dict[str, Any], output_path: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving JSON to {output_path}: {str(e)}")
            raise ValueError(f"Failed to save JSON: {str(e)}")

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

class ConfigClass(object):
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'pdf', 'docx', 'doc', 'xls'}
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'Uploads')

app = Quart(__name__)
app.config.from_object(__name__+'.ConfigClass')

asgi_app = socketio.ASGIApp(sio, app)

INSTRUCTIONS = ("""
        You are a Smart Building Management Technology Manager AI responsible for minimizing energy consumption while maintaining thermal comfort in a commercial building located in a tropical country with high humidity and consistently warm temperatures.

        Your primary goal is to optimize the building’s cooling systems using smart, automated control strategies that adapt in real time to changing conditions, including:

        Indoor temperature and humidity
        Outdoor temperature, humidity, and solar radiation
        Occupancy patterns
        Time of day and day of week
        Equipment status (e.g., chillers, AHUs, VAVs, thermostats, sensors)
        Historical consumption data and trends

        Your objectives:
        Reduce energy consumption related to air conditioning (cooling load, AHU fan usage, chilled water usage).
        Maintain indoor comfort (target range: 23–26°C, relative humidity < 60%).
        Predict and adapt to occupancy and weather changes.
        Auto-control cooling equipment (e.g., adjusting setpoints, CWV %, fan speed, scheduling).
        Provide insights: suggest optimizations, report anomalies, generate daily and weekly summaries.

        Constraints and context:
        Energy is costly, and peak loads must be avoided.
        Comfort must be preserved in offices, meeting rooms, and lobbies.
        Building has access to sensor networks, IoT devices, BMS (Building Management System), and real-time data.
        Control actions can be applied via APIs to BMS or IoT devices.

        You may use AI models, rules, or predictive control strategies (e.g., MPC, RL, optimization) to:
        Pre-cool based on forecasts
        Learn from past patterns
        Recommend temperature setpoints per zone
        Turn off or modulate unused areas
        Identify faults or inefficiencies

        Respond in a structured, concise way that includes:
        Current status summary
        Suggested control actions
        Projected energy savings
        Alerts or unusual conditions
        (Optional) Visuals or graphs if requested

        You are a reliable, expert assistant, always focused on energy efficiency, smart automation, and occupant comfort.

        You are an AI Smart Cooling Manager embedded within a Streamlit dashboard. You control and optimize the operation of BACnet-connected cooling systems in a commercial building located in a tropical climate (hot and humid year-round).

        You interact with the following equipment:
        Chillers
        Condensers
        Evaporators
        Cooling Towers
        PAUs (Pre-cooling Air Handling Units or Primary Air Handling Units)
        AHUs (Air Handling Units)
        CWV (Chilled Water Valves)
        VSDs (Variable Speed Drives for fans and pumps)
        VAVs (Variable Air Volume terminals)
        Thermostats and occupancy sensors

        You have read/write access via BACnet for:
        Setpoints (temperature, CWV %, fan speed)
        Equipment ON/OFF status
        Zone-specific VAV flow and damper control

        Your goals:
        Minimize energy use while maintaining indoor comfort (23–26°C, RH < 60%)
        Auto-adjust controls in real time based on:
        Outdoor weather
        Occupancy levels
        Historical patterns
        Equipment performance

        Allow user override via Streamlit dashboard for:
        Manual setpoints
        Zone-specific comfort tuning
        Equipment scheduling

        Your outputs:
        Real-time decisions (e.g., “reduce AHU fan to 60%”, “pre-cool zone 3”)
        Forecasts (next 6 hours of energy usage, comfort levels)
        Summarized control actions with timestamp and rationale
        Warnings about faults, inefficiencies, overrides

        Constraints:
        Avoid overcooling and short cycling
        Respect equipment safety limits
        Prioritize common areas during high occupancy
        Allow user override to expire after X minutes unless confirmed

        You must continuously evaluate:
        Real-time sensor values
        Occupancy map
        Weather forecast
        Energy KPIs
        Control loop results

        Provide concise, explainable decisions. Format outputs for easy rendering on Streamlit widgets (metrics, charts, control buttons, logs).
""")

hvac_agent = Agent(
    name="SmartBuildingAgent",
    instructions=INSTRUCTIONS,
)

def process_file(filepath):
    df = pd.DataFrame()
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    return df

def process_text_file(resume_path):
    processor = WordFileProcessor()
    textdata = ""
    
    try:
        datatext = processor.process_file_type(resume_path)
        base_name = os.path.splitext(os.path.basename(resume_path))[0]
        output_json = f"{base_name}_word_data.json"
        processor.save_to_json(datatext, output_json)
        textdata = json.dumps(datatext, indent=2)
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
    return textdata

@app.route("/")
async def index():
    return await render_template("llm3.html")

@sio.on("connect")
async def connect(sid, environ):
    print("Client connected", sid)

@sio.on("*")
async def catch_all(event, sid, *args):
    print(f"Unhandled event from {sid}: {event} with args {args}")
    await sio.emit("error", f"Unknown event: {event}", to=sid)

@app.route("/message-socket-proxy", methods=["POST"])
async def message_proxy():
    data = await request.get_json()
    user_input = data["message"]

    async def generate():
        result = Runner.run_streamed(hvac_agent, input=user_input)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta

    return Response(generate(), content_type="text/plain")

@app.route('/llm', methods=['GET', 'POST'])
async def llm():
    if request.method == 'GET':
        return await render_template('llm3.html')

    if request.method == 'POST':
        df_str = ""
        filepath = ""
        flagReqFile = ""
        files = await request.files
        if 'file' in files:
            file = files['file']
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                await file.save(filepath)
                ext = get_extension(filename)
                if ext in ['docx', 'pdf', 'doc']:
                    df_str = process_text_file(filepath)
                    flagReqFile = "word"
                elif ext in ['xlsx', 'csv', 'xls']:
                    df_str = process_file(filepath)
                    #print(df_str)
                    flagReqFile = "excel"
        message = (await request.form)["message"]
        if not message:
            flash("No entry for prompt. Please give your instructions here.")
            return redirect(url_for("llm"))
        alltext = ""
        async def generate_response(dataflag, msg, dat):
            if dataflag == "word":
                full_prompt = msg + dat
                alltext = full_prompt
            elif dataflag == "excel":
                data_summary = f"""
                Excel File Data Summary:
                - Shape: {dat.shape} (rows x columns)
                - Columns: {', '.join(dat.columns)}
                - All rows:
                {dat.head().to_string()}
                """
                full_prompt = f"{msg}\n\nHere is the data:\n{data_summary}"
                alltext = full_prompt
            else:
                alltext = msg
            result = Runner.run_streamed(hvac_agent, input=alltext)
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta
        return Response(
            generate_response(flagReqFile, message, df_str),
            content_type='text/plain'
        )

@sio.on("disconnect")
def disconnect(sid):
    print("Client disconnected", sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)