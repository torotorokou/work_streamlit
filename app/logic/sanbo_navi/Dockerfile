FROM python:3.10-slim

WORKDIR /app

COPY scr/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY scr/ /app/scr

CMD ["uvicorn", "scr.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]