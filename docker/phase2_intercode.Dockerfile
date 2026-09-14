FROM ubuntu:22.04

ARG FS_VERSION=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash bsdmainutils cpio coreutils cron curl diffutils dnsutils findutils gawk git grep \
    gzip imagemagick iproute2 iputils-ping jq md5deep ncompress net-tools procps psmisc \
    python3 rename sed tar tree util-linux && \
    rm -rf /var/lib/apt/lists/*

COPY docker/bash_scripts/setup_nl2b_fs_${FS_VERSION}.sh /opt/intercode/setup.sh
RUN chmod +x /opt/intercode/setup.sh && /opt/intercode/setup.sh

# InterCode evaluates filesystem effects with `git status --short`. Its image
# ignores operating-system paths and commits the task filesystem as the clean
# episode state.
COPY docker/docker.gitignore /.gitignore
RUN git config --global user.email "intercode@pnlp.org" && \
    git config --global user.name "intercode" && \
    git init / && \
    git -C / add -A && \
    git -C / commit -m "initial InterCode filesystem"

LABEL org.opencontainers.image.source="https://github.com/princeton-nlp/intercode"
LABEL org.opencontainers.image.revision="c3e46d827cfc9d4c704ec078f7abf9f41e3191d8"

WORKDIR /
CMD ["sleep", "infinity"]
