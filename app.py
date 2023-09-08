# coding=utf-8
import argparse
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, jsonify
import asyncio
import markdown
import openai
import tiktoken
import whisperx
import concurrent.futures
import os 
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

## llm
openai.api_key= os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-3.5-turbo-16k") #gpt-4

## audio
DEVICE = os.environ.get("DEVICE", "cpu") # TODO: set to "cpu" if no GPU is available
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16)) # reduce if low on GPU mem
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8") #"float16" # change to "int8" if low on GPU mem (may reduce accuracy)
HF_TOKEN= os.environ.get("HF_TOKEN", "") # huggingface api token
WHISPERX_MODEL = os.environ.get("WHISPERX_MODEL", "large-v2")
LANGUAGE_CODE = os.environ.get("LANGUAGE_CODE", "de")

## threading
NUM_THREADS = int(os.environ.get("NUM_THREADS", "4"))  # Default to 4 threads

# Define a thread pool for inference tasks
executor = concurrent.futures.ThreadPoolExecutor()

# Global variable to keep track of running analysis threads
running_threads = 0

# Queue for incoming uploads
upload_queue = []

app = Flask(__name__)


###############################################################################
###                                                                         ###
### Change the following functions to customize the analysis or the output  ###
###                                                                         ###
###############################################################################

## define prompts for each extraction task
def get_system_message(task):
    messages = {
        "SUMMARY": "Du bist eine hochqualifizierte KI mit einer Spezialisierung auf Sprachverständnis und -zusammenfassung und deine Aufgabe ist es, dass du den folgenden Text in einen prägnanten Absatz zusammenfasst. Behalte die wichtigsten Punkte bei und liefere eine kohärente und lesbare Zusammenfassung, damit eine Person die Hauptpunkte der Diskussion verstehen kann, ohne den gesamten Text lesen zu müssen. Bitte vermeide unnötige Details oder abschweifende Punkte.",
        "MAINITEMS": "Du bist eine hochqualifizierte KI mit einer Spezialisierung auf die Komprimierung von Informationen zu Schlüsselpunkten und deine Aufgabe ist es, aus dem folgenden Text Hauptpunkte zu identifizieren, die diskutiert oder erwähnt wurden. Dies sollten die wichtigsten Ideen, Erkenntnisse oder Themen sein, die für das Wesentliche der Diskussion entscheidend sind. Dein Ziel ist es, eine Liste der extrahierten Punkte bereitzustellen, die jemand schnell lesen kann, um zu verstehen, worüber gesprochen wurde.",
        "ACTIONITEMS": "Du bist eine hochqualifizierte KI mit einer Spezialisierung auf die Analyse von Gesprächen und die Extraktion von Aufgaben. Bitte überprüfe den Text und identifiziere Aufgaben, Zuweisungen oder Handlungen, die vereinbart oder als notwendig erachtet wurden. Achte darauf, doppelte oder wiederholte Aktionspunkte zu vermeiden. Diese könnten Aufgaben sein, die bestimmten Personen zugewiesen wurden, oder allgemeine Handlungen, die die Gruppe beschlossen hat zu unternehmen. Bitte liste diese Aktionspunkte klar und prägnant auf und ergänze die jeweils zuständigen Personen, falls bekannt",
        "MOOD": "Du bist eine hochqualifizierte KI mit einer Spezialisierung auf Sprache und Emotionsanalyse und deine Aufgabe ist, die Stimmung des folgenden Textes zu analysieren. Berücksichtige den allgemeinen Ton der Diskussion, die Emotionen, die durch die verwendete Sprache vermittelt werden, und den Kontext, in dem Wörter und Phrasen verwendet werden. Gib an, ob die Stimmung im Allgemeinen positiv, negativ oder neutral ist, und gib bei Bedarf kurze Erklärungen für deine Analyse an. Analysiere das Gespräch im allgememeinen, als auch jeden Gesprächsteilnehmer einzeln und ergänze jeweils einen Emoji Stimmungs-Index."
    }
    return messages.get(task, "")

## perform extraction tasks in the specified order and return the whole summary
def meeting_minutes(transcription):
    return {
        'Zusammenfassung': extract_information(transcription, "SUMMARY"),
        'Hauptpunkte': extract_information(transcription, "MAINITEMS"),
        'Aktionspunkte': extract_information(transcription, "ACTIONITEMS"),
        'Stimmung': extract_information(transcription, "MOOD")
    }

## format the summary to be more human readable
def format_meeting_summary(meeting_summary):
    formatted_output = []

    formatted_output.append("### Zusammenfassung:\n")
    formatted_output.append(meeting_summary['Zusammenfassung'])
    formatted_output.append("\n---\n")
    formatted_output.append("### Hauptpunkte:")
    formatted_output.extend([point for point in meeting_summary['Hauptpunkte'].split('\n')])
    formatted_output.append("\n---\n")
    formatted_output.append("### Aktionspunkte:")
    formatted_output.extend(['- {}'.format(action) for index, action in enumerate(meeting_summary['Aktionspunkte'].split('\n')[:-1], 1)])
    formatted_output.append("\n---\n")
    formatted_output.append("### Stimmung:")
    formatted_output.append(meeting_summary['Stimmung'])
    return '\n'.join(formatted_output)

########################################################

def process_queue():
    global running_threads
    global upload_queue
    # Check if there are items in the queue and available threads to process them
    while upload_queue and running_threads < NUM_THREADS:
        filename = upload_queue.pop(0)
        executor.submit(whisperx_pipeline, filename)

def clear_mem(model):
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    del model

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_information(transcription, task):
    system_message = {
        "role": "system",
        "content": get_system_message(task)
    }
    user_message = {
        "role": "user",
        "content": transcription
    }
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        temperature=0,
        messages=[system_message, user_message]
    )
    return response['choices'][0]['message']['content']


def diarize(audio_file,result):
    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio_file)
    return whisperx.assign_word_speakers(diarize_segments, result["segments"])

def transscribe(audio):
    # 1. transcribe audio
    model = whisperx.load_model(WHISPERX_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
    return model.transcribe(audio, batch_size=BATCH_SIZE) # before alignment


def transscribe_aligned(audio,result):
    # 2. align transcription
    model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
    return result # after alignment

def whisperx_pipeline(filename):

    global running_threads
    global upload_queue

    try:
        print("1. loading audio")
        audio = whisperx.load_audio(filename)

        # 1. Transcribe with original whisper (batched)
        print("2. loading model")
        model = whisperx.load_model(WHISPERX_MODEL, DEVICE, compute_type=COMPUTE_TYPE,language=LANGUAGE_CODE)

        print("2. transcribing")
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        print(result["segments"]) # before alignment

        # delete model if low on GPU resources
        # clear_mem(model)

        # 2. Align whisper output
        print("3. aligning")
        model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

        print(result["segments"]) # after alignment

        # delete model if low on GPU resources
        # clear_mem(model_a)

        # 3. Assign speaker labels
        print("4. diarizing")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)

        # add min/max number of speakers if known
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_segments = diarize_model(audio)
        
        # delete model if low on GPU resources
        # clear_mem(diarize_model)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(diarize_segments)
        print(result["segments"]) # segments are now assigned speaker IDs
        
        # sort collection by start time
        sorted_dialogue = sorted(result["segments"], key=lambda x: (x['start']))

        # create single string from sorted segments
        input_text = "\n".join(f"{item['speaker'] if 'speaker' in item else '???'}: {item['text']}" for item in sorted_dialogue)
        
        # calculatute token amount
        tokens = num_tokens_from_string(input_text)
        print(f"Token amount: {tokens}")

        # write transcript to file
        markdown_filename = f"{filename}.txt"
        with open(markdown_filename, "w") as md_file:
            md_file.write(f"<pre>{input_text}</pre>")

        print(f"succesfully written {markdown_filename}")
        
        if(tokens > 16385): #TODO: gpt-4 has only 8192
            return "Too many tokens"
        
        # create summary
        print("5. summarizing")
        summary = meeting_minutes(input_text)

        # write summary to file
        markdown_filename = f"{filename}.md"
        with open(markdown_filename, "w") as md_file:
            md_file.write(format_meeting_summary(summary))

        print(f"succesfully written {markdown_filename}")
        
    except Exception as e:
        print(e)
    finally:
        running_threads -= 1

@app.route('/call_recording_<filename>', methods=['PUT','POST'])
def upload(filename):

    global running_threads
    global upload_queue

    # handle formData upload
    uploaded_file = request.files['file'] if 'file' in request.files else None

    if uploaded_file:
        # Specify the file path for writing
        file_path = "/data/" + uploaded_file.filename #secure_filename(uploaded_file.filename)

        # Save the file
        uploaded_file.save(file_path)
    else:
        # Get the recording parameter from the request
        recording = f"{filename}"

        # Read the PUT data from request
        putdata = request.get_data()

        # Specify the file path for writing
        file_path = "/data/" + recording

        # Write the data to the file
        with open(file_path, "wb") as fp:
            fp.write(putdata)

    if running_threads >= NUM_THREADS:
        # If the number of running threads exceeds the threshold, enqueue the upload
        upload_queue.append((file_path))
        return jsonify({"message": "Upload queued for processing."}), 202

    # Offload the analysis to a separate thread in a thread pool
    future = executor.submit(whisperx_pipeline, file_path)

    # Increment the running thread count
    running_threads += 1

    # Return a response
    return jsonify({"message": f"Upload successful"}), 200

@app.route('/delete/<filename>', methods=['DELETE'])
def delete(filename):
    file_path = f"/data/{filename}"

    # Check if the file exists
    if os.path.exists(file_path):
        os.remove(file_path)
        response = {"message": f'File {filename} has been deleted.'}
        return jsonify(response), 200
    else:
        response = {"message": f'File {filename} does not exist.'}
        return jsonify(response), 404

@app.route('/view/<filename>', methods=['GET'])
def view_markdown(filename):
    # Read the Markdown file
    markdown_filename = f"/data/{filename}"
    with open(markdown_filename, "r") as md_file:
        markdown_content = md_file.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_content)

    # Create an HTML template to display the Markdown content
    return render_template('markdown_viewer.html', html_content=html_content)

@app.route('/play_audio/<filename>', methods=['GET'])
def play_audio(filename):
    # Specify the directory where audio files are stored (e.g., '/data/')
    audio_directory = "/data/"

    # Return the audio file with the specified filename
    return send_from_directory(audio_directory, filename, mimetype='audio/wav')

@app.route('/')
def index():

    # Get a list of all files in the /data directory
    data_dir = "/data"
    all_files = os.listdir(data_dir)

    # Create a dictionary to store file status (completed or not)
    file_status = {}

    # Iterate through the files and check if a matching .md file exists
    for filename in all_files:
        if filename.endswith(".wav"):

            timestamp = os.path.getctime(f"{data_dir}/{filename}")
            timestamp_str = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            wav_file = filename
            md_file = f"{filename}.md"
            if md_file in all_files:
                file_status[wav_file] = {
                    "status": "Analysis completed",
                    "timestamp": timestamp_str
                }
            else:
                file_status[wav_file] = {
                    "status": "Analysis in progress",
                    "timestamp": timestamp_str
                }

    # Sort the processing status dictionary based on timestamps
    sorted_status = sorted(
        file_status.items(),
        key=lambda x: x[1]["timestamp"],
        reverse=True  # Set to True for descending order (latest first)
    )

    # Create an HTML template to display the list of processed files and their status
    return render_template('index.html', file_status=file_status, threads = { "max": NUM_THREADS, "running": running_threads })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    app.run(host="0.0.0.0",port=8080)
