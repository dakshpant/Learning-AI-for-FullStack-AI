import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


JD = """
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""
RESUME = """
DAKSH PANT
+91-9599030493 | dakshpant15@gmail.com | github.com/dakshpant | linkedin.com/in/dakshpant | LeetCode: 100+ problems solved
PROFESSIONAL SUMMARY
Backend-focused Software Engineer with hands-on experience building production SaaS applications using Node.js, Fastify, TypeScript, PostgreSQL, MongoDB, Redis, and Keycloak. Experienced in designing secure RESTful APIs, database-driven business workflows, and role-based access control (RBAC) systems. Contributed across the full development lifecycle, including authentication, authorization, backend architecture, and frontend integration for production platforms.
TECHNICAL SKILLS
Languages: JavaScript, TypeScript, C++, Python, SQL
Backend Development: Node.js, Express.js, Fastify, REST APIs, JWT, Zod, Microservices
Frontend Development: React.js, EJS, HTML, CSS, Bootstrap, Tailwind CSS
Databases & ORMs: PostgreSQL, MySQL, MongoDB, Redis, Prisma, Sequelize, Mongoose
Security & IAM: Keycloak, Authentication, Authorization, RBAC, JWT Validation (jwks-rsa)
Tools & Practices: Git, GitHub, Docker, Postman, ClickUp, Agile, CI/CD, Code Review, UAT
Core Computer Science: Data Structures & Algorithms, OOP, DBMS, Operating Systems, Computer Networks
PROFESSIONAL EXPERIENCE
Software Development Intern, Buzybug (formerly VMS Techs)	July 2025 – Present
Noida, Uttar Pradesh, India	
Backend Developer Intern  (Jan 2026 – Present)
	•	Developed scalable backend services and 25+ RESTful APIs for production SaaS applications using Node.js, Fastify, Prisma, PostgreSQL, MongoDB, and Redis, supporting restaurant management and HR platforms.
	•	Engineered secure authentication and authorization by integrating Keycloak, JWT validation (jwks-rsa), and database-driven RBAC, securing 12+ REST APIs across 5 organizational roles.
	•	Built backend workflows for restaurant onboarding, menu management, QR code provisioning, analytics, and bulk CSV imports, covering REST API design, business logic, validation, and frontend-backend integration.
	•	Built reusable backend components including request validation with Zod, authentication middleware, file upload handling, structured logging, and modular service architecture, improving code consistency across services.
	•	Contributed 100+ Git commits while participating in peer code reviews, UAT testing, and production release cycles following Agile and Git-based development workflows.
Frontend Developer Intern  (Jul 2025 – Dec 2025)
	•	Built responsive React-based interfaces and analytics dashboards while contributing to authentication APIs and database-driven features using Sequelize and PostgreSQL.
PROJECTS
Qrtli — SaaS QR Management & Restaurant Platform	
Node.js, PostgreSQL, MongoDB, Redis, EJS	
	•	Developed backend features across QR Management, Store Management, QR Menu, and URL Shortening by implementing REST APIs, business logic, database workflows, and third-party integrations.
	•	Implemented restaurant onboarding workflows supporting configurable menu management, bulk CSV import, duplicate validation, image uploads, and automatic QR provisioning.
	•	Integrated Redis-backed URL workflows and MongoDB analytics to visualize QR scans, shortened URL activity, browser insights, and customer engagement through interactive dashboards.
	•	Production Deployment: qrtli.com
WizSuit HR SuperAdmin
Node.js, Fastify, Keycloak, PostgreSQL	
	•	Developed backend services for a multi-tenant HR platform implementing authentication, user lifecycle management, organization administration, profile management, and role administration using Fastify and PostgreSQL.
	•	Built a database-driven RBAC system enabling configurable module-level permissions across frontend navigation and backend APIs.
	•	Integrated Keycloak Admin APIs for onboarding, authentication, password management, and role administration using JWT validation (jwks-rsa).
EDUCATION
Guru Gobind Singh Indraprastha University, Delhi	2022 – 2026
Bachelor of Technology in Computer Science and Engineering — CGPA: 8.10/10	
CERTIFICATIONS & ACHIEVEMENTS
	•	Solved 100+ problems on LeetCode as part of a structured, pattern-based DSA learning curriculum.
	•	Maintain "Learn-DSA-WITH-PATTERNS," a public GitHub repository documenting pattern-based DSA problem-solving.
	•	Completed Claude 101 certification (Anthropic).

"""


def ask_llm(system_prompt, user_prompt):
    sys_msg = {
        "role": "system",
        "content": system_prompt
    }
    user_msg = {
        "role": "user",
        "content": user_prompt
    }
    messages = [sys_msg, user_msg]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def step1_res_extract(RESUME):
    # extract resume skills
    system_prompt = """ You are a professional HR assistant. Extract The skill sform the candidates resume provided, only return the skills and no other information and do not invent and add new skills by yourself."""
    user_prompt = f"Extract the skills form this resume: {RESUME}"

    return ask_llm(system_prompt, user_prompt)


def step2_JD_extract(JD):
    print("step2")
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the Job description  provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt = f"""
    Extract the skills from this JD
    {JD}
    """

    return ask_llm(system_prompt, user_prompt)


def step3_match(candidate, jd):
    print("step3")
    system_prompt = """
    You are a professional HR assistant. compare the skills of candidate and the skills required in the JD and produce a final score between
    1 and 100. also produce a short verdict whther the candidate is a good fit for the role.
    """
    user_prompt = f"""
    Compare and matc h the skills
    JD:
    {jd}
    Candidate:
    {candidate}
    """

    return ask_llm(system_prompt, user_prompt)


candidate = step1_res_extract(RESUME)
print(candidate)
sleep(2)
jd = step2_JD_extract(JD)
print(jd)
sleep(2)
score = step3_match(candidate, jd)
print(score)
