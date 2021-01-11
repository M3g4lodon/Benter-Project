.PHONY: downgrade_remove_generate_upgrade_migration
downgrade_remove_generate_upgrade_migration:
	PYTHONPATH=. ENVIRONMENT=development alembic downgrade base
	rm -r ./alembic/versions/*
	PYTHONPATH=. ENVIRONMENT=development alembic revision -m  "create_tables" --autogenerate
	PYTHONPATH=. ENVIRONMENT=development alembic upgrade head

.PHONY: run_db
run_db:
	docker-compose up --remove-orphans --build

.PHONY: jupyter_lab
jupyter_lab:
	jupyter lab

.PHONY: vulture
vulture:
	vulture --sort-by-size --exclude tests,alembic .
