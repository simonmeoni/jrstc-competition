FROM pytorch/pytorch:latest
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
RUN mkdir "jrstc-competition"
COPY . .