#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = capstone-stock-prediction
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if conda info --envs | grep -q $(PROJECT_NAME); then \
		echo '>>> Conda environment $(PROJECT_NAME) already exists.'; \
	else \
		conda create -n $(PROJECT_NAME) python=$(PYTHON_VERSION) -y; \
		echo '>>> New conda environment created. Activate with:\nconda activate $(PROJECT_NAME)'; \
	fi"
	
## Run the sentiment analysis
.PHONY: sentiment_analysis
sentiment_analysis:
	$(PYTHON_INTERPRETER) sentiment_analysis/main.py


## Process dataset with arguments
.PHONY: data
data:
	$(PYTHON_INTERPRETER) stock_prediction/dataset.py
	$(PYTHON_INTERPRETER) stock_prediction/features.py

## EDA
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) stock_prediction/plots.py

## Run training and prediction for the four architectures
.PHONY: run_all
run_all:
	sh scripts/script.sh

## Making trading simulation
.PHONY: simulate
simulate:
	sh scripts/trading.sh

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
