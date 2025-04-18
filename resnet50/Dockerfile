FROM --platform=${TARGETPLATFORM:-linux/amd64} python:3.13-slim AS build

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN addgroup --system app \
    && adduser app --system --ingroup app --home /home/app \
    && chown app:app /home/app

USER app
ENV APP_HOME=/home/app/

WORKDIR $APP_HOME

COPY --chown=app:app requirements.txt .

USER root
RUN pip install --no-cache-dir -r requirements.txt

FROM build AS ship
ENV PORT=8080

WORKDIR $APP_HOME


COPY . ./

CMD ["sh", "-c", "gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 'app:createApp()'"]
