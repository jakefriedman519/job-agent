import dataclasses
import pandas as pd

CSV_FILEPATH = 'scraped_jobs.csv'

@dataclasses.dataclass
class Job:
    id: int
    company: str
    title: str
    location: str
    description: str
    url: str


def job_to_string(job: Job) -> str:
    return f'Job ID: {job.id}\nCompany: {job.company}\nTitle: {job.title}\nLocation: {job.location}' \
           f'\nDescription: {job.description}\nURL: {job.url}'


def get_id_from_job_string(job_str: str) -> int:
    return int(job_str.split('\n')[0][8:])


def parse_job(job: pd.Series) -> Job:
    job_dict = job.to_dict()
    return Job(
        id=job_dict['ID'],
        company=job_dict['Company Name'],
        title=job_dict['Job Title'],
        location=job_dict['Location'],
        description=job_dict['Job Description'],
        url=job_dict['url'] + '/apply'
    )


job_df = pd.read_csv(CSV_FILEPATH)
job_listings = []
job_listings_strings = []
for i in range(len(job_df)):
    job = job_df.iloc[i]
    job_listing = parse_job(job)
    job_listings.append(job_listing)
    job_listings_strings.append(job_to_string(job_listing))