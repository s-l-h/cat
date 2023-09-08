# 🐈 CAT 

CAT (**C**onversation **A**nalytics **T**oolkit) is a versatile tool for analyzing and summarizing audio conversations. It utilizes [`Whisper-X`](https://github.com/m-bain/whisperX), to transcribe and diarize spoken content. After that, it performs analytic tasks on this transcript, using a large language model (currently GPT-only). With CAT, you can quickly extract key information, action items, and even analyze the mood of the conversation.

## Features

- **Audio Transcription**: CAT can transcribe audio files, making them accessible and searchable as text.

- **Diarization**: The tool can identify and label speakers in a conversation, providing insights into who said what.

- **Information Extraction**: CAT extracts important information, including summaries, key points, and action items, from the transcribed content.

- **Emotion Analysis**: It also offers sentiment and mood analysis, helping you gauge the overall tone of the conversation.

- **Web Interface**: CAT provides a user-friendly web interface for easy interaction and analysis.

- **REST API**: CAT includes a REST endpoint that allows you to programmatically upload audio files for automatic analysis. This feature is particularly useful when integrated with VoIP systems like Asterisk or FreeSWITCH for automatic call analytics.

## Getting Started

### ❗ Prerequisites:

- [OpenAI API Key](https://beta.openai.com/account/api-keys)
- [HuggingFace API Token](https://huggingface.co/settings/token)
- Accepted EULA for:
  - [pyannote.segmentation](https://huggingface.co/pyannote/segmentation)
  - [pyannote.VAD](https://huggingface.co/pyannote/voice-activity-detection)
  - [pyannote.diarize](https://huggingface.co/pyannote/speaker-diarization)

### Running Locally

To run CAT locally, follow these steps:

1. Clone this repository:

   ```shell
   git clone https://github.com/yourusername/CAT.git
   cd CAT
   ```

2. Set up the environment variables by creating a `.env` file in the root directory:

   ```shell
   OPENAI_API_KEY=your_openai_api_key
   GPT_MODEL=gpt-3.5-turbo-16k
   DEVICE=cpu  # Change to "cuda" if you have a GPU
   BATCH_SIZE=16
   COMPUTE_TYPE=int8  # Change to "float16" if needed
   HF_TOKEN=your_huggingface_api_token
   WHISPERX_MODEL=large-v2
   LANGUAGE_CODE=de  # Change to your desired language code
   NUM_THREADS=4
   ```

   Replace `your_openai_api_key` and `your_huggingface_api_token` with your actual API keys.

3. Install the required Python packages:

   ```shell
   pip install -r requirements.txt
   ```

4. Start the 🐈 server:

   ```shell
   python app.py
   ```

5. Access the web interface at `http://localhost:8080` in your web browser.

### 🐳 Running in a Docker Container

This repo also provides Docker containers for both CPU and GPU environments. Here's how to run 🐈 within a container:

#### ♿ CPU-Only Container

To run CAT in a CPU-only Docker container, use the following steps:

1. Build the Docker image:

   ```shell
   docker build -t cat-cpu -f Dockerfile .
   ```

2. Run the Docker container:

   ```shell
   docker run -d --env-file=.env -p 8080:8080 -v /path/to/your/audio/files:/data cat-cpu
   ```

#### 💉 GPU-Accelerated Container

To run CAT in a GPU-accelerated Docker container, use the following steps:

1. Make sure, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on the host.
   

2. Build the Docker image:

   ```shell
   docker build -t cat-gpu -f Dockerfile.gpu .
   ```

3. Run the Docker container with GPU support:

   ```shell
   docker run --gpus all -d --env-file=.env -p 8080:8080 -v /path/to/your/audio/files:/data cat-gpu
   ```

Replace `/path/to/your/audio/files` with the directory where your audio files are located.



## Usage

1. **Web Interface**: You can use the web interface to manually upload and analyze audio recordings. This is the easiest way to get started with 🐈.

2. **REST API**: CAT offers a REST API endpoint that allows you to programmatically upload audio files for automatic analysis. To use this feature, you can send a `POST` or `PUT` request to the `/` endpoint. The uploaded audio will be processed automatically, and the analysis results will be made available via the web interface. This is particularly useful when integrating CAT with VoIP systems for automatic call analytics
<br/>   

   Here's an example of how to use the REST API endpoint:

    ```shell
    # Replace <filename> with your desired filename
    curl -X POST -F "file=@/path/to/your/audio/file.wav" http://localhost:8080/<filename>
    ```
 .

1. **Monitor Progress**: CAT will process the uploaded audio in an async manner and display its progress on the web interface. You can check the status and see how many analytic threads are running.

2. **View Results**: Once processing is complete, you can view the transcribed text, diarized conversation, and extracted information. CAT also provides a summary of the conversation's key points and mood analysis.

3. **Download**: You can download the transcribed text and summary for your records.

## Customize Analysis

You can customize the analysis and output by modifying the relevant functions in the `app.py` file, as explained in the README.

## Dependencies

CAT relies on the following technologies and libraries:

- [OpenAI GPT](https://beta.openai.com/): For text generation and summarization.

- [Whisper-X](https://github.com/m-bain/whisperX): For audio transcription and diarization.

- Flask: For the web application framework.

- Other Python libraries as listed in `requirements.txt`.

## Contributing

Feel free to contribute to 🐈 by creating issues or submitting pull requests. Your contributions are welcome and appreciated.

## License

🐈 is open-source software licensed under the MIT License. See the [LICENSE](LICENSE) file for details.