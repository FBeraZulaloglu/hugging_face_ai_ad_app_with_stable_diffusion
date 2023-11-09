FROM python:3.8-slim

# Set the working directory to /create_ai_ad
WORKDIR /create_ai_ad

# Copy the current directory contents into the container at /create_ai_ad
COPY . /create_ai_ad

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# To get permission to the user
Run useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
      PATH=/home/user/.local/bin:$PATH

# Make port 80 available to the world outside this container
EXPOSE 80

ENV NAME StableAd

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
