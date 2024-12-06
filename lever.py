from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By  # Import By
from selenium.webdriver.common.keys import Keys
import time
from constants import COVER_LETTER
from selenium.webdriver.common.action_chains import ActionChains
from utils import setup_driver  # to sleep
import os

# Fill in this dictionary with your personal details!
JOB_APP = {
    "first_name": "Jake",
    "last_name": "Friedman",
    "pronouns": "He/him",
    "email": "jakefriedman519@gmail.com",
    "phone": "201-744-9353",
    "org": "Northeastern",
    "resume": "resume.pdf",
    "resume_textfile": "",
    "linkedin": "https://www.linkedin.com/in/jacob-friedman-779644217/",
    "website": "https://github.com/jakefriedman519",
    "github": "https://github.com/jakefriedman519",
    "twitter": "",
    "other_website": "",
    "location": "Boston, Massachusetts, United States",
    "grad_month": '05',
    "grad_year": '2025',
    "university": "Northeastern University",
    "cover_letter": COVER_LETTER
}

def lever(driver):
    # navigate to the application page
    #driver.find_element(By.CLASS_NAME, 'template-btn-submit').click()

    # basic info
    first_name = JOB_APP['first_name']
    last_name = JOB_APP['last_name']
    full_name = first_name + ' ' + last_name  # f-string should work here
    driver.find_element(By.NAME, 'name').send_keys(full_name)
    driver.find_element(By.NAME, 'email').send_keys(JOB_APP['email'])
    driver.find_element(By.NAME, 'phone').send_keys(JOB_APP['phone'])
    driver.find_element(By.NAME, 'org').send_keys(JOB_APP['org'])
    time.sleep(1)

    # socials
    driver.find_element(By.NAME, 'urls[LinkedIn]').send_keys(JOB_APP['linkedin'])
    try:  # try both versions
        driver.find_element(By.NAME, 'urls[Github]').send_keys(JOB_APP['github'])
    except NoSuchElementException:
        try:
            driver.find_element(By.NAME, 'urls[GitHub]').send_keys(JOB_APP['github'])
        except NoSuchElementException:
            pass
    try:
        driver.find_element(By.NAME, 'urls[Portfolio]').send_keys(JOB_APP['website'])
    except NoSuchElementException:
        pass
    try:
        driver.find_element(By.NAME, 'urls[Twitter]').send_keys(JOB_APP['twitter'])
    except NoSuchElementException:
        pass
    try:
        driver.find_element(By.NAME, 'urls[Other]').send_keys(JOB_APP['other_website'])
    except NoSuchElementException:
        pass

    time.sleep(1)

    # add university
    try:
        driver.find_element(By.CLASS_NAME, 'application-university').click()
        search = driver.find_element(By.XPATH, "//*[@type='search']")
        search.send_keys(JOB_APP['university'])  # find university in dropdown
        search.send_keys(Keys.RETURN)
    except NoSuchElementException:
        pass
    time.sleep(1)

    # other field fills...
    # Same structure as above, use By.NAME, By.XPATH, By.CLASS_NAME, etc. as needed for each field.
    driver.find_element(By.NAME, 'comments').send_keys(JOB_APP['cover_letter'])

    try:
        # Scroll to the checkbox
        checkbox = driver.find_element(By.XPATH, f"//input[@name='pronouns' and @value='{JOB_APP['pronouns']}']")
        ActionChains(driver).move_to_element(checkbox).perform()

        # Now click the checkbox
        checkbox.click()
    except NoSuchElementException:
        pass

    # resume upload
    resume_path = os.path.abspath(JOB_APP['resume'])
    driver.find_element(By.NAME, 'resume').send_keys(resume_path)  # replace with the correct path
    time.sleep(5)

def apply(url):
    driver = setup_driver()
    driver.get(url)
    if 'lever' in url:
        lever(driver)
    else:
        print('Error: Lever job application not found')
    # driver.quit()
