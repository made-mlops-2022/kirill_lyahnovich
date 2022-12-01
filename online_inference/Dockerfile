FROM python:3.9.15-slim-bullseye
COPY . .

RUN pip install --no-cache-dir  --upgrade pip
RUN pip install --no-cache-dir --upgrade -r  requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.run_service:app", "--host", "0.0.0.0",  "--port", "8000"]