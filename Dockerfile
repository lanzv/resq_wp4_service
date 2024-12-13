FROM ubuntu:22.04

# Perform the basic setup
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get upgrade -y
RUN apt-get install wget git -y

# Install Python (miniconda) and create a new environment
RUN mkdir -p /opt/miniconda3 &&\
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh &&\
	bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 &&\
	rm -rf /opt/miniconda3/miniconda.sh
RUN /opt/miniconda3/bin/conda upgrade -y conda && /opt/miniconda3/bin/conda create -y --name pyserv python=3.10
SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "pyserv", "/bin/bash", "-c"]

# Requires a NVIDIA GPU with Compute Capability >= 7.5, see https://developer.nvidia.com/cuda-gpus
RUN conda install pytorch=2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia




# Copy project files
COPY ./resources /opt/resq_wp4_service/resources
COPY ./src /opt/resq_wp4_service/src
COPY ./app.py /opt/resq_wp4_service/app.py
COPY ./gunicorn_logging.conf /opt/resq_wp4_service/logging.conf
COPY ./requirements.txt /opt/resq_wp4_service/requirements.txt

# Set working directory
WORKDIR /opt/resq_wp4_service


RUN mkdir -p /opt/saved_models

# Create the target directory
RUN mkdir -p /opt/saved_models/evidence_extraction

# Download all files and save them to /opt/saved_models/evidence_extraction
RUN wget -P /opt/saved_models/evidence_extraction \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/model.safetensors" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/special_tokens_map.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/tokenizer.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/tokenizer_config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/evidence_extraction/vocab.txt"


# Create the target directory
RUN mkdir -p /opt/saved_models/answer_prediction

# Download all files and save them to /opt/saved_models/answer_prediction
RUN wget -P /opt/saved_models/answer_prediction \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/generation_config.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/model.safetensors" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/special_tokens_map.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/tokenizer.json" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/tokenizer.model" \
    "https://ufallab.ms.mff.cuni.cz/~lanz/answer_prediction/tokenizer_config.json"



# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt



# Create the entrypoint script file and add the content
RUN echo '#!/bin/bash\n' \
    '# Script that supervisor uses to keep the resq_wp4_service running.\n' \
    'if ! ps ax | grep -v grep | grep "gunicorn app:app --timeout 0 --bind 0.0.0.0:8081 --worker-class uvicorn.workers.UvicornWorker" > /dev/null\n' \
    'then\n' \
    '    # Log restart\n' \
    '    echo "Resq WP4 service down; restarting run_resq_wp4_service.sh"\n' \
    '    # Run the gunicorn service\n' \
    '    gunicorn app:app --timeout 0 --bind 0.0.0.0:8081 --worker-class uvicorn.workers.UvicornWorker --log-config /opt/resq_wp4_service/logging.conf --log-level info\n' \
    'fi\n' \
    | sed 's/^ //g' \
    > /opt/run_resq_wp4_service.sh

RUN chmod +x /opt/run_resq_wp4_service.sh

	
# Install Supervisord
RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Supervisor configuration
RUN echo '[program:run_resq_wp4_service]\n' \
    'command=/opt/run_resq_wp4_service.sh\n' \
    'autostart=true\n' \
    'autorestart=true\n' \
    'stderr_logfile=/var/log/run_resq_wp4_service.err.log\n' \
    'stdout_logfile=/var/log/run_resq_wp4_service.out.log\n' \
    | sed 's/^ //g' \
    > "/etc/supervisor/conf.d/supervisord.conf"


# Expose port 8081
EXPOSE 8081

CMD ["/usr/bin/supervisord","-n", "-c", "/etc/supervisor/supervisord.conf"]