# Stage 1: Build and setup environment
FROM nvidia/cuda:12.3.0-runtime-ubuntu20.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 git build-essential cmake libreadline-dev libncurses5-dev zlib1g-dev libssl-dev \
    curl ca-certificates && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN /opt/conda/bin/conda init bash
RUN conda create -n resq_wp4_service python=3.11 && conda clean -afy

# Activate environment and install dependencies
COPY ./requirements.txt .
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate resq_wp4_service && pip install -r requirements.txt"

# Download all models and files
RUN mkdir -p /opt/models/evidence_extraction && \
    wget -P /opt/models/evidence_extraction \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/model.safetensors" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/special_tokens_map.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/tokenizer.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/tokenizer_config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/vocab.txt"

RUN mkdir -p /opt/models/answer_prediction && \
    wget -P /opt/models/answer_prediction \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/generation_config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/model.safetensors" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/special_tokens_map.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/tokenizer.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/tokenizer.model" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/tokenizer_config.json"

# Stage 2: Runtime image
FROM nvidia/cuda:12.3.0-runtime-ubuntu20.04

# Copy files from build stage
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/models /opt/models
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda
RUN /opt/conda/bin/conda init bash

# Copy project files
COPY ./resources /opt/resq_wp4_service/resources
COPY ./src /opt/resq_wp4_service/src
COPY ./app.py /opt/resq_wp4_service/app.py
COPY ./gunicorn_logging.conf /opt/resq_wp4_service/logging.conf

# Set working directory
WORKDIR /opt/resq_wp4_service

# Install Supervisord
RUN apt-get update && apt-get install -y --no-install-recommends supervisor procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Supervisord configuration
RUN echo '[program:run_resq_wp4_service]\n' \
    'command=/opt/run_resq_wp4_service.sh\n' \
    'autostart=true\n' \
    'autorestart=true\n' \
    'stderr_logfile=/var/log/run_resq_wp4_service.err.log\n' \
    'stdout_logfile=/var/log/run_resq_wp4_service.out.log\n' \
    | sed 's/^ //g' \
    > "/etc/supervisor/conf.d/supervisord.conf"

# Create the entrypoint script file and add the content
RUN echo '#!/bin/bash\n' \
    '# Script that supervisor uses to keep the RES-Q WP4 Service running.\n' \
    '. /opt/conda/etc/profile.d/conda.sh\n' \
    'if ! ps ax | grep -v grep | grep "gunicorn app:app --timeout 0 --bind 0.0.0.0:8081 --worker-class uvicorn.workers.UvicornWorker" > /dev/null\n' \
    'then\n' \
    '    # Log restart\n' \
    '    echo "RES-Q WP4 Service down; restarting run_resq_wp4_service.sh"\n' \
    '    # The right conda environment\n' \
    '    conda activate resq_wp4_service\n' \
    '    # Run the Django application using gunicorn\n' \
    '    gunicorn app:app --timeout 0 --bind 0.0.0.0:8081 --worker-class uvicorn.workers.UvicornWorker\n' \
    'fi\n' \
    | sed 's/^ //g' \
    > /opt/run_resq_wp4_service.sh
RUN chmod +x /opt/run_resq_wp4_service.sh

# Expose port
EXPOSE 8081

# Start the app
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]