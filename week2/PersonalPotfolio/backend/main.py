from fastapi import FastAPI
import os
import json
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

model = "openai/gpt-oss-120b"
app = FastAPI()

# resume parser


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: int | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []
    socials: list[str] = []


resume_schema = Resume.model_json_schema()


class chat_request(BaseModel):
    ques: str


def ask_candidate(ques: str, resume: Resume):

    system_prompt = f"""
    You are an Ai assistennt representing a job candidate.
    Below is everything you know about the candidate.
    {resume.model_dump_json(indent=2)}

    RULES:
    1. Answer only using this information.

    2. Never hallucinate.

    3. If information is unavailable,
    say

    "I don't have enough information to answer that."

    4. Be professional.
    5. Answer as if HR is interviewing this candidate.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                # the user's question
                "role": "user",
                "content": ques
            }

        ],
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            yield content


def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    message_system = {
        "role": "system",
        "content": system_prompt
    }
    message_user = {
        "role": "user",
        "content": user_prompt
    }
    messages = [message_system, message_user]
    response_format = {
        "type": "json_object"
    }
    response = client.chat.completions.create(
        model=model, messages=messages, response_format=response_format, temperature=0)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


# pdf extraction
def read_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


@app.get("/")
def home():
    resume_text = read_pdf(Path("Daksh Pant CV .pdf"))
    resume = parse_resume(resume_text)
    print(resume.model_dump_json(indent=2))
    return {
        "message": "Running FastAPI"
    }


resume = None


@app.post("/chat")
def chat(request: chat_request):
    global resume
    if resume is None:
        resume_text = read_pdf(Path("Daksh Pant CV .pdf"))
        resume = parse_resume(resume_text)

    answer = ask_candidate(request.ques, resume)
    return StreamingResponse(
        ask_candidate(request.ques, resume),
        media_type="text/plain"
    )
