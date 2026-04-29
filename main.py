import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

# -------------------------------
# CONFIG
# -------------------------------
API_URL = "https://jsearch.p.rapidapi.com/search"

HEADERS = {
    "X-RapidAPI-Key": os.getenv("X-RAPIDAPI-KEY"),
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------
# FASTAPI INIT
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# REQUEST MODEL
# -------------------------------
class JobRequest(BaseModel):
    skills: str
    country: str = "us"


# -------------------------------
# BUILD SEARCH QUERY
# -------------------------------
def build_search_query(user_skills):
    skills = [s.strip() for s in user_skills.split(",") if s.strip()]

    if len(skills) >= 2:
        return f"{skills[0]} {skills[1]}"
    elif len(skills) == 1:
        return skills[0]
    else:
        return "software"


# -------------------------------
# FETCH JOBS
# -------------------------------
def fetch_jobs(user_skills, country):
    try:
        query = build_search_query(user_skills)

        params = {
            "query": query,
            "page": "1",
            "num_pages": "2",
            "country": country
        }

        response = requests.get(
            API_URL,
            headers=HEADERS,
            params=params,
            timeout=20
        )

        print("STATUS CODE:", response.status_code)
        print("SEARCH QUERY:", query)
        print("COUNTRY:", country)
        print("FULL API RESPONSE:")
        print(response.text)

        data = response.json()

        return data.get("data", [])

    except Exception as e:
        print("FETCH ERROR:", str(e))
        return []

# -------------------------------
# AI SCORE
# -------------------------------
def semantic_score_sync(user_skills, title, description):
    try:
        prompt = f"""
User skills:
{user_skills}

Job title:
{title}

Job description:
{description}

Give a job match score from 0 to 100.
Return only the number.
"""

        res = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return int(res.output_text.strip())

    except Exception as e:
        print("AI SCORE ERROR:", str(e))
        return 0


# -------------------------------
# AI EXPLANATION
# -------------------------------
def explain_match_sync(user_skills, title, description):
    try:
        prompt = f"""
User skills:
{user_skills}

Job title:
{title}

Job description:
{description}

Explain in one short sentence why this job matches.
"""

        res = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return res.output_text.strip()

    except Exception as e:
        print("AI EXPLAIN ERROR:", str(e))
        return "Could not generate explanation."


# -------------------------------
# PROCESS SINGLE JOB
# -------------------------------
async def process_job(job, user_skills):
    title = job.get("job_title", "")
    description = job.get("job_description", "")
    company = job.get("employer_name", "")
    location = job.get("job_city", "")
    link = job.get("job_apply_link", "")

    try:
        score_task = asyncio.to_thread(
            semantic_score_sync,
            user_skills,
            title,
            description
        )

        explain_task = asyncio.to_thread(
            explain_match_sync,
            user_skills,
            title,
            description
        )

        score, explanation = await asyncio.gather(
            score_task,
            explain_task
        )

    except Exception as e:
        print("PROCESS ERROR:", str(e))
        score = 0
        explanation = "Unable to process."

    return {
        "title": title,
        "company": company,
        "location": location,
        "score": score,
        "why_match": explanation,
        "apply_link": link
    }


# -------------------------------
# FIND JOBS
# -------------------------------
async def find_jobs_async(user_skills, country):
    jobs = fetch_jobs(user_skills, country)

    jobs = [j for j in jobs if j.get("job_apply_link")]

    jobs = jobs[:8]

    tasks = [
        process_job(job, user_skills)
        for job in jobs
    ]

    results = await asyncio.gather(*tasks)

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# -------------------------------
# API ENDPOINT
# -------------------------------
@app.post("/jobs")
async def get_jobs(req: JobRequest):
    try:
        results = await find_jobs_async(
            req.skills,
            req.country
        )

        return {
            "status": "success",
            "user_skills": req.skills,
            "country": req.country,
            "jobs": results
        }

    except Exception as e:
        print("ENDPOINT ERROR:", str(e))

        return {
            "status": "error",
            "message": str(e),
            "jobs": []
        }


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "AI Job API running successfully"
    }