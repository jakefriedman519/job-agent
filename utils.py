import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import logging
import os

def setup_driver():
    options = Options()
    options.add_experimental_option('detach', True)
    options.add_argument("--start-maximized")
    service = Service('/usr/local/bin/chromedriver')
    return webdriver.Chrome(options=options)

def load_user_data():
    with open("config/user_data.json", "r") as f:
        return json.load(f)

def log_message(message):
    logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info(message)
