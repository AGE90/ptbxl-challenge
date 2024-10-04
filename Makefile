setup:  ## Setup the environment.
    bash scripts/setup_env.sh

test:   ## Run all tests.
    pytest tests/

run:    ## Run the application.
    python app/main.py
