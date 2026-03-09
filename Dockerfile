# Finance Dashboard: Streamlit app + ETL
FROM python:3.11-slim

WORKDIR /workspace

# Optional system deps for weasyprint (uncomment if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
COPY configs ./configs
COPY etl ./etl
COPY qa ./qa
COPY pyproject.toml *.md ./

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
