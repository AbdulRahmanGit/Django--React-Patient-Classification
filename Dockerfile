FROM ghcr.io/puppeteer/puppeteer:17.0.0
COPY . /home/pptruser/src
ENTRYPOINT [ "/bin/bash", "-c", "node -e \"$(</home/pptruser/src/probe.js)\"" ]