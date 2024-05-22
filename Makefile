.PHONY: run setup
TARGET = main.py


setup: requirements.txt
	pip install -r requirements.txt
	pip install -r requirements_test.txt


run:
	python $(TARGET)

run_all:
	python load_data.py csv
	python eda.py
	python main.py

format_and_clean:
	black $(TARGET)
	isort $(TARGET)
	unimport $(TARGET)
	ruff $(TARGET)

clean_pycache:
	rm -rf __pycache__