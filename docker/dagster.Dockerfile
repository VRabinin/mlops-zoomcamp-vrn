FROM python:3.9.12-slim

RUN mkdir -p /opt/dagster/dagster_home /opt/dagster/app

RUN pip install dagster-webserver dagster-postgres dagster-aws

ARG CACHE_DATE=not_set
# Copy your code and workspace to /opt/dagster/app
COPY 03-orchestration/dagster_conf/requirements.txt /opt/dagster/app/
# Set the environment variable DAGSTER_HOME to /opt/dagster/dagster_home/
WORKDIR /opt/dagster/app
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ /opt/dagster/app/data/

COPY 03-orchestration/pipeline/repo.py 03-orchestration/pipeline/workspace.yaml /opt/dagster/app/
ENV DAGSTER_HOME=/opt/dagster/dagster_home/

# Copy dagster instance YAML to $DAGSTER_HOME
COPY 03-orchestration/dagster_conf/dagster.yaml /opt/dagster/dagster_home/
# Set the working directory
WORKDIR /opt/dagster/app

EXPOSE 3000

#ENTRYPOINT ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000"]
ENTRYPOINT ["dagster", "dev", "-h", "0.0.0.0", "-p", "3000"]