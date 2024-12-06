# Job Application Chat Agent

### Overview

This project is an LLM-based chat agent used for searching for and applying to jobs.
A sample dataset of 100 computing-related jobs were scraped from Lever for this project.

### Usage

**Note**: The code in this repo has been tested with Python 3.10

```shell
pip install -r requirements.txt
python agent.py
```

This will launch the Gradio interface on localhost, which you can use through your browser.

#### Credits
The job dataset was scraped from Lever's job board. The scraper used can be found [here](https://github.com/ghiarishi/job-scraper/).