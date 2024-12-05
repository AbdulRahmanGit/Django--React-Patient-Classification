FROM ghcr.io/puppeteer/puppeteer:17.0.0

# Copy the entire contents of the current directory to /home/pptruser/src/streamlit in the container
COPY . /home/pptruser/src/streamlit

# Set the working directory to the location where the probe.js file is located
WORKDIR /home/pptruser/src/streamlit/probe-action

# Set the entry point to execute the probe.js script with Node.js
ENTRYPOINT [ "/bin/bash", "-c", "node -e \"$(</home/pptruser/src/probe.js)\"" ]
