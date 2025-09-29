FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


##
##FROM python:3.9-slim

##WORKDIR /app

##COPY requirements.txt .
##RUN pip install --no-cache-dir -r requirements.txt

##COPY . .

##EXPOSE 8000

##CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]