FROM openjdk:11-jre-slim

# Versions of different tools installed.
ENV NODEJS_VERSION="v8.9.3"
ENV BAZEL_VERSION="0.8.0"
ENV ANDROID_TOOLS_VERSION="3859397"

# Update package lists and install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    software-properties-common \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    curl \
    libicu-dev \
    g++ \
    git \
    python \
    python-dev \
    python-setuptools \
    unzip \
    nano \
    automake \
    bc \
    libc-dev \
    libreadline-dev \
    libtool \
    make \
    ncurses-dev \
    nvi \
    pkg-config \
    python \
    python-dev \
    python-setuptools \
    unzip \
    wavpack \
    zip \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Clone language resources
WORKDIR /usr/local/src
RUN git clone https://github.com/wannaphongcom/language-resources

# Load language resources python modules
ENV PYTHONPATH="/usr/local/src/language-resources"

# Install python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
      && bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user \
      && rm bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# Add bazel to path
ENV PATH=/root/bin:$PATH

# Install Nodejs
WORKDIR /opt/
RUN wget http://storage.googleapis.com/gae_node_packages/node-${NODEJS_VERSION}-linux-x64.tar.gz \
      && tar -xf node-${NODEJS_VERSION}-linux-x64.tar.gz \
      && rm node-${NODEJS_VERSION}-linux-x64.tar.gz

ENV PATH $PATH:/opt/node-${NODEJS_VERSION}-linux-x64/bin

# Fetch and prepare Festival & friends
WORKDIR /usr/local/src/tools
ENV FESTIVAL_SUIT_PATH /usr/local/src/tools
RUN /usr/local/src/language-resources/festival_utils/setup_festival.sh

RUN rm /usr/local/src/tools/festival/lib/festival.el

# Set env variables
ENV ESTDIR /usr/local/src/tools/speech_tools
ENV FESTVOXDIR /usr/local/src/tools/festvox
ENV FESTIVALDIR /usr/local/src/tools/festival
ENV FLITEDIR /usr/local/src/tools/flite
ENV SPTKDIR /usr/local/src/tools/
ENV EXDIR /usr/local/src/tools/examples

# Add festival to path
ENV PATH="/usr/local/src/tools/festival/bin:${PATH}"

WORKDIR /usr/local/src/language-resources
COPY th.zip /usr/local/src/language-resources/th.zip
RUN unzip th.zip
COPY wavs.zip /usr/local/src/language-resources/wavs.zip
RUN unzip wavs.zip
ENV WAV_FOLDER /usr/local/src/language-resources/wavs
RUN mkdir tts
ENV OUTPUT_VOICE_FOLDER /usr/local/src/language-resources/tts
