FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY coretex/ ./coretex/
COPY modules/ ./modules/
COPY distributions/ ./distributions/

CMD ["uvicorn", "distributions.cortx.main:app", "--host", "0.0.0.0", "--port", "8000"]
