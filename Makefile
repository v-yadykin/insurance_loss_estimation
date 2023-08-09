up:
	docker compose up -d

up-test:
	cp test.env .env
	docker compose up -d

down:
	docker compose down --remove-orphans

restart-app:
	docker compose restart app celery

recreate-app:
	docker compose up --force-recreate app celery -d

build:
	docker compose build
