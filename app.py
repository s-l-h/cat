from flask import Flask, request

app = Flask(__name__)

@app.route('/call_recording_<filename>.<extension>', methods=['PUT'])
def upload(filename,extension):
    # Get the recording parameter from the request
    recording = f"{filename}.{extension}"

    # Read the PUT data from request
    putdata = request.get_data()

    # Specify the file path for writing
    file_path = "/data/" + recording

    # Write the data to the file
    with open(file_path, "wb") as fp:
        fp.write(putdata)

    # Return a response
    return 'File saved at ' + file_path

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
