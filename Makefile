# Convenience targets; everything maps onto plain `docker compose` commands.
SHELL := /bin/bash
COMPOSE ?= docker compose
PY ?= python3

.PHONY: env build up down stop restart-data-node logs ps flower venv test clean-results

env: .env

.env: .env.example
	cp .env.example .env
	echo "" >> .env
	echo "AIRFLOW_UID=$$(id -u)" >> .env
	echo "FERNET_KEY=$$($(PY) -c 'import base64,secrets;print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())')" >> .env
	echo "AIRFLOW__API_AUTH__JWT_SECRET=$$($(PY) -c 'import secrets;print(secrets.token_urlsafe(48))')" >> .env
	@echo "wrote .env"

build: env
	$(COMPOSE) build

up: env
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

stop:
	$(COMPOSE) stop

# After changing DATA_NODE_MEM_LIMIT in .env, only this node needs recreating.
restart-data-node: env
	$(COMPOSE) up -d --force-recreate data-node

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

flower: env
	$(COMPOSE) --profile flower up -d flower

venv:
	$(PY) -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e '.[dev]'

test:
	./venv/bin/pytest -q

clean-results:
	rm -rf data/results/*
	touch data/results/.gitkeep
