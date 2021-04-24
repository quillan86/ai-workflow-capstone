# https://sourcery.ai/blog/python-docker/
# https://medium.com/swlh/alpine-slim-stretch-buster-jessie-bullseye-bookworm-what-are-the-differences-in-docker-62171ed4531d
# use a python 3.8 image
FROM python:3.8-buster AS base

# sets the environmental variable and reads an existing environment variable if one exists to that
# like to do this even if repetitive because it's very clear then what it is
# can set defaults if you need
# docker-related
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
# etl-specific related
#
# make this python-deps
FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install --upgrade pip
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install python dependencies in /.venv
COPY Pipfile .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv update
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

FROM base AS runtime

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# expose port
EXPOSE 80

# Install application into container
COPY ./src/app ./app

# run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
