# Paper Orchestra Frontend

This directory contains the Streamlit frontend for Paper Orchestra.

## How to Launch the Demo Locally

1. Ensure you have activated your environment and installed dependencies.
2. Navigate to the project root directory:
   ```bash
   cd paper-orchestra
   conda activate paper_orchestra
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run frontend/app.py
   ```
4. Open the URL provided in the terminal (usually `http://localhost:8501` or similar) in your browser.

## How to Launch the Demo in a Linux Container

From the project root, create the output directory and start the container:

```bash
mkdir -p demo_logs
DOCKER_UID="$(id -u)" DOCKER_GID="$(id -g)" docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). The container runs Linux, including when Docker Desktop is used on macOS. Create a root `.env` file with the API credentials needed by the selected model; Compose passes it to the running container, and `.dockerignore` keeps it out of the image. Generated run data is available in the root `demo_logs/` directory.

Stop the demo with `Ctrl+C`, or run `docker compose down` from the project root.

## How to Kill Unfinished Writing Processes

The frontend launches a background process (`paper_writing_cli.py`) to handle the paper generation. 

### Automatic Cleanup
On Linux, the application is configured to automatically kill the background process if the Streamlit server process is stopped. Simply pressing **Ctrl+C** in the terminal where you ran `streamlit run` should clean up the running pipeline.

### Manual Cleanup
If you disconnect from the frontend or the server stops unexpectedly without cleaning up, you may have orphan processes running in the background. You can kill them manually using:

```bash
pkill -f paper_writing_cli.py
```

Or to find the process ID (PID) first:
```bash
pgrep -f paper_writing_cli.py
```
And then kill it:
```bash
kill <PID>
```
