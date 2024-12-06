from openai import OpenAI
from typing import List, Optional
from datetime import date, time, datetime
import dataclasses
import gradio as gr
import yaml
from pathlib import Path
import rag
from lever import apply
import jobs as j
from jobs import Job

API_KEY = 'friedman.ja@northeastern.edu:lFhgheytZKAV10WNFoep'
BASE_URL = 'http://199.94.61.113:8000/v1/'

"""
resp = client.chat.completions.create(
    messages = [{
        "role": "user",
        "content": "Write short complaint to The Boston Globe about the rat problem at Northeastern CS. Blame the math department. No more than 4 sentences."
    }],
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0)
"""

SYSTEM_PROMPT = """
You are a helpful job application agent. Respond to queries with JUST ONE Python block that
uses one of the following functions:

def find_jobs(self, query: str) -> List[Job]:
    ...
    
# job_id should be the ID of a Job
def apply_to_job(self, job_id: int) -> Optional[int]:
    ...

Return the result in a variable called result. Don't use list comprehensions.
Return the function used in a variable called tool.

If the user is not trying to apply to a job or find jobs, simply return your text response in result.
If the user wants to apply to a job by ID, use apply_to_job(id)

Today's date is September 1 2023.
"""

FEW_SHOT_RESPONSE_ONE = """
```python
result = find_jobs("Get me all the jobs in San Francisco")
tool = "find_jobs"
```
"""

FEW_SHOT_RESPONSE_TWO = """
```python
result = apply_to_job(1)
tool = "apply_to_job"
```
"""

FEW_SHOT_RESPONSE_THREE = """
```python
result = find_jobs("Find me all the software engineer jobs")
tool = "find_jobs"
```
"""

FEW_SHOT_RESPONSE_FOUR = """
```python
result = apply_to_job(result[0].id)
tool = "apply_to_job"
```
"""

FEW_SHOT_RESPONSE_FIVE = """
```python
result = "You probably want a job in software engineering or a related field!"
tool = None
```
"""


@dataclasses.dataclass
class AgentResponse:
    """
 The superclass for all agent responses.
 """
    text: str

@dataclasses.dataclass
class FindJobsResponse(AgentResponse):
    """
 The agent used the `find_jobs` tool and found the following jobs.
 """
    available_jobs: List[Job]


@dataclasses.dataclass
class ApplyToJobResponse(AgentResponse):
    """
 The agent used the `apply_to_job` tool and applied to the following job.
 """
    applied_job: Optional[int]


@dataclasses.dataclass
class TextResponse(AgentResponse):
    pass


def extract_code(resp_text):
    code_start = resp_text.find("`")
    code_end = resp_text.rfind("`")
    if code_start == -1 or code_end == -1:
        return "pass"
    return resp_text[code_start + 3 + 7:code_end - 2]


def get_text_from_applied_job(job_id: Optional[int]):
    if job_id is not None:
        return f"I've filled out job {job_id} for you! Check the new tab for any required information I may have missed."
    else:
        return "I couldn't apply to that job for you."


class Agent:
    # The complete conversation with the LLM, including the system prompt.
    conversation: List[dict]
    display_history: List[dict]
    # The formatted response from the last tool call.
    text_prefix: Optional[str]
    # The current database of jobs. The tools update this database.
    jobs: List[Job]
    client: OpenAI
    # Global variables used in tool calls.
    program_state: dict

    def __init__(self):
        self.conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": "Get me all the jobs in San Francisco"
            },
            {
                "role": "assistant",
                "content": FEW_SHOT_RESPONSE_ONE
            },
            {
                "role": "user",
                "content": "Apply to the job with ID 1."
            },
            {
                "role": "assistant",
                "content": FEW_SHOT_RESPONSE_TWO
            },
            {
                "role": "user",
                "content": "Find me all the software engineer jobs"
            },
            {
                "role": "assistant",
                "content": FEW_SHOT_RESPONSE_THREE
            },
            {
                "role": "user",
                "content": "Apply to the first one"
            },
            {
                "role": "assistant",
                "content": FEW_SHOT_RESPONSE_FOUR
            },
            {
                "role": "user",
                "content": "I'm a student studying computer science, what kinds of jobs should I look for?"
            },
            {
                "role": "assistant",
                "content": FEW_SHOT_RESPONSE_FIVE
            },
        ]
        self.jobs = j.job_listings
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.program_state = {
            "find_jobs": self.find_jobs,
            "apply_to_job": self.apply_to_job,
            "result": None,
            "date": date,
            "tool": None
        }
        self.display_history = []
        self.text_prefix = ""

    def find_jobs(self, query: str) -> List[Job]:
        relevant_job_ids = rag.fetch_relevant_jobs(query)
        found_jobs = []
        for job_id in relevant_job_ids:
            found_jobs.append(j.job_listings[job_id])
        return found_jobs

    def apply_to_job(self, job_id: int) -> Optional[int]:
        job = self.get_job_by_id(job_id)
        if job is not None:
            apply(job.url)
            return job_id
        return None

    def get_text_from_job_list(self, job_list):
        text = ""
        if isinstance(job_list, list) and len(job_list) > 0:
            text = "Here are the jobs I found:"
            for job in job_list:
                text += "\n\n"
                text += j.job_to_string(job)

        else:
            text = "I couldn't find any matching flights. Please be sure to include a date and origin/destination"
        return text

    def say(self, user_message: str) -> AgentResponse:
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        self.display_history.append({
            "role": "user",
            "content": user_message
        })
        resp = self.client.chat.completions.create(messages=self.conversation,
                                                   model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                   temperature=0)

        resp_text = resp.choices[0].message.content
        self.conversation.append({
            "role": "assistant",
            "content": resp_text
        })
        code_text = extract_code(resp_text)
        agent_response = self.run_code(code_text)
        return agent_response

    def run_code(self, code_text):
        try:
            exec(code_text, self.program_state)
        except Exception as e:
            print(e)
        if self.program_state["tool"] == 'find_jobs':
            self.text_prefix = "Finding jobs...\n\n"
            response = FindJobsResponse(
                text=self.get_text_from_job_list(self.program_state["result"]),
                available_jobs=self.program_state["result"])
        elif self.program_state["tool"] == 'apply_to_job':
            self.text_prefix = "Applying to job...\n\n"
            response = ApplyToJobResponse(text=get_text_from_applied_job(self.program_state["result"]),
                                          applied_job=self.program_state["result"])
        else:
            self.text_prefix = ""
            if self.program_state["result"] is not None:
                response = TextResponse(
                    text=self.program_state["result"])
            else:
                response = TextResponse(text="Something went wrong")
        self.display_history.append({"role": "assistant", "content": self.text_prefix + response.text})
        return response
    
    def get_job_by_id(self, job_id):
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None


class EvaluationResult:
    score: float
    conversation: List[dict]

    def __init__(self, score, conversation):
        self.score = score
        self.conversation = conversation


def eval_agent(benchmark_file: str) -> EvaluationResult:
    """
    Evaluate the agent on the given benchmark YAML file.
    """
    agent = Agent()  # Initialize the agent
    with open(benchmark_file, "r") as file:
        steps = yaml.safe_load(file)

    for n, step in enumerate(steps):
        response = agent.say(step["prompt"])
        match step["expected_type"]:
            case "text":
                if not isinstance(response, TextResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "find-jobs":
                if not isinstance(response, FindJobsResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
                found_keyword = False
                for job in response.available_jobs:
                    for keyword in step['expected_keywords']:
                        if keyword in j.job_to_string(job).lower():
                            found_keyword = True
                            break
                if not found_keyword:
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "apply-to-job":
                if not isinstance(response, ApplyToJobResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.applied_job != step["expected_result"]:
                    return EvaluationResult(n / len(steps), agent.conversation)
    return EvaluationResult(1.0, agent.conversation)


def eval_all_benchmarks() -> float:
    all_benchmarks = [str(p) for p in Path(".").glob("*.yaml")]  # Load all YAML benchmark files
    scores = []
    for benchmark_file in all_benchmarks:
        print(f"Running benchmark: {benchmark_file}")
        print("====================================\n")
        eval_response = eval_agent(benchmark_file)
        scores.append(eval_response.score)
        print(f'Score: {eval_response.score}')
        if eval_response.score != 1:
            print(f'Conversation: {eval_response.conversation}\n')
    return sum(scores) / len(scores)


def job_agent(agent, user_message):
    agent.say(user_message)
    conversation = "\n\n".join(
        [f"{turn['role'].capitalize()}:\n{turn['content']}" for turn in agent.display_history]
    )
    return conversation


# This is if you want to run the GUI demo
if __name__ == "__main__":
    agent = Agent()
    with gr.Blocks() as gui:
        conversation_history = gr.Textbox(label="Conversation History", lines=10, interactive=False, autoscroll=True)
        user_input = gr.Textbox(label="Your Message", lines=1)

        # Button to send message and update conversation
        submit = gr.Button("Send")
        submit.click(fn=lambda user_message: job_agent(agent, user_message),
                     inputs=user_input, outputs=conversation_history)

        gui.launch()
