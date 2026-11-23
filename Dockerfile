FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    texlive-fonts-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app/paper-orchestra

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
