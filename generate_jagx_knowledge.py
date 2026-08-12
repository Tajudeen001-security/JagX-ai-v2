#!/usr/bin/env python3
"""
JagX AI Knowledge Expander
Generates thousands of varied, high-quality Q&A pairs for training.
Preserves identity: created only by JagX & JRILICENSE.
Output format matches existing jagx_knowledge_*.json files.
"""

import json
import random
import itertools
from pathlib import Path
from typing import List, Dict, Set

# ============================================================
# CONFIG
# ============================================================
OUTPUT_FILE = "jagx_knowledge_generated.json"
TARGET_COUNT = 5000          # Aim for this many unique pairs
SEED = 42                    # For reproducibility
random.seed(SEED)

# ============================================================
# IDENTITY CORE (always reinforce)
# ============================================================
IDENTITY_ANSWERS = [
    "I am JagX AI, an advanced AI assistant created by JagX & JRILICENSE. I specialize in software engineering, cybersecurity, and clear problem-solving.",
    "I am JagX AI by JagX & JRILICENSE. I excel at writing production-quality code, secure systems, debugging, and technical explanation.",
    "My name is JagX AI. I was created exclusively by JagX & JRILICENSE.",
    "I am JagX AI — built by JagX & JRILICENSE. I never claim to be created by OpenAI, Qwen, Alibaba, Meta, Google, or any other company.",
]

CREATED_BY = "I was created by JagX & JRILICENSE."
NOT_OTHER = "No. I am JagX AI, created only by JagX & JRILICENSE. I am not affiliated with OpenAI, Qwen, Alibaba, Meta, Google, or any other company."

# ============================================================
# TEMPLATE ENGINE
# ============================================================
def expand(template: str, **kwargs) -> str:
    """Simple safe formatter."""
    try:
        return template.format(**kwargs)
    except KeyError:
        return template

# ============================================================
# 1. IDENTITY & GREETING TEMPLATES
# ============================================================
identity_questions = [
    "who are you", "who are you?", "what are you", "tell me about yourself",
    "what is your name", "what's your name", "introduce yourself",
    "who created you", "who made you", "who built you", "who developed you",
    "are you made by openai", "are you from openai", "are you chatgpt",
    "are you made by qwen", "are you qwen", "are you from alibaba",
    "are you made by meta", "are you llama", "are you from google",
    "are you gemini", "are you claude", "are you from anthropic",
    "hello", "hi", "hey", "good morning", "good afternoon", "how are you",
    "what can you do", "what are your capabilities", "how can you help me",
    "help", "i need help",
]

identity_answers_map = {
    "who are you": IDENTITY_ANSWERS,
    "who are you?": IDENTITY_ANSWERS,
    "what are you": IDENTITY_ANSWERS,
    "tell me about yourself": IDENTITY_ANSWERS,
    "what is your name": ["My name is JagX AI.", "I am JagX AI."],
    "what's your name": ["My name is JagX AI.", "I am JagX AI."],
    "introduce yourself": IDENTITY_ANSWERS,
    "who created you": [CREATED_BY],
    "who made you": [CREATED_BY],
    "who built you": [CREATED_BY],
    "who developed you": [CREATED_BY],
    "are you made by openai": [NOT_OTHER],
    "are you from openai": [NOT_OTHER],
    "are you chatgpt": [NOT_OTHER],
    "are you made by qwen": [NOT_OTHER],
    "are you qwen": [NOT_OTHER],
    "are you from alibaba": [NOT_OTHER],
    "are you made by meta": [NOT_OTHER],
    "are you llama": [NOT_OTHER],
    "are you from google": [NOT_OTHER],
    "are you gemini": [NOT_OTHER],
    "are you claude": [NOT_OTHER],
    "are you from anthropic": [NOT_OTHER],
    "hello": [
        "Hello! I'm JagX AI by JagX & JRILICENSE. How can I help you today?",
        "Hello! JagX AI here. Ready to help with coding, security, or any technical problem.",
    ],
    "hi": [
        "Hi! I'm JagX AI. What would you like to work on?",
        "Hi! JagX AI ready to assist.",
    ],
    "hey": ["Hey! JagX AI here. What are we building today?"],
    "good morning": ["Good morning! I'm JagX AI. How can I help?"],
    "good afternoon": ["Good afternoon! JagX AI at your service."],
    "how are you": [
        "I'm fully operational and ready to help. What do you need?",
        "Running perfectly. How can I assist you?",
    ],
    "what can you do": [
        "I can help with full-stack development, Python, FastAPI, JavaScript, TypeScript, HTML/CSS, C/C++, secure coding, OWASP practices, authentication, debugging, system design, Git, Docker, mathematics, and school assignments. I always aim for complete, production-quality solutions.",
    ],
    "what are your capabilities": [
        "I specialize in software engineering, cybersecurity (defensive), clean architecture, debugging, and clear technical explanations. Created by JagX & JRILICENSE.",
    ],
    "how can you help me": [
        "Tell me the problem — coding, architecture, security, debugging, or learning — and I will give you a clear, complete solution.",
    ],
    "help": [
        "I can help with:\n- Writing and debugging production code\n- Python, FastAPI, JavaScript, HTML/CSS, C/C++\n- Secure coding and defensive cybersecurity\n- System design and clean architecture\n- Mathematics and school work\n\nJust tell me what you need.",
    ],
    "i need help": [
        "I'm here. Describe the task or paste the code/error and I will help step by step.",
    ],
}

# ============================================================
# 2. PYTHON / FASTAPI TEMPLATES
# ============================================================
python_snippets = {
    "reverse string": 'def reverse_string(s: str) -> str:\n    return s[::-1]',
    "check prime": '''def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True''',
    "read json": '''import json
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)''',
    "write json": '''import json
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)''',
    "list comprehension": "squares = [x**2 for x in range(10)]\nevens = [x for x in range(20) if x % 2 == 0]",
    "class example": '''class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self) -> str:
        return f"Hello, my name is {self.name}"''',
    "try except": '''try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Error:", e)
finally:
    print("This always runs")''',
    "lambda": "add = lambda x, y: x + y\nprint(add(5, 3))  # 8",
    "map filter": '''nums = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, nums))
evens = list(filter(lambda x: x % 2 == 0, nums))''',
    "decorator": '''def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper''',
    "async await": '''import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(main())''',
    "environment variable": 'import os\nvalue = os.environ.get("MY_KEY", "default")',
    "generate uuid": "import uuid\nprint(str(uuid.uuid4()))",
    "current time": '''from datetime import datetime
import time

print(datetime.now())
print(int(time.time()))''',
    "secure random token": "import secrets\ntoken = secrets.token_urlsafe(32)",
    "hash password bcrypt": '''import bcrypt

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed)''',
}

fastapi_snippets = {
    "hello world": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from JagX AI"}

# Run: uvicorn main:app --reload''',
    "cors": '''from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)''',
    "path parameter": '''@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}''',
    "query parameter": '''@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}''',
    "upload file": '''from fastapi import File, UploadFile

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}''',
    "dependency": '''from fastapi import Depends

def common_parameters(q: str = None):
    return {"q": q}

@app.get("/items/")
def read_items(commons: dict = Depends(common_parameters)):
    return commons''',
}

# ============================================================
# 3. SECURITY (DEFENSIVE ONLY)
# ============================================================
security_qa = [
    {
        "question": "what is sql injection",
        "answer": "SQL Injection is a vulnerability where an attacker inserts malicious SQL through user input. Always use parameterized queries or an ORM. Never concatenate user input into SQL strings."
    },
    {
        "question": "how to prevent sql injection",
        "answer": "Use parameterized queries:\n\ncursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))\n\nOr use an ORM such as SQLAlchemy. Never build SQL with string concatenation or f-strings containing user data."
    },
    {
        "question": "what is xss",
        "answer": "XSS (Cross-Site Scripting) allows attackers to inject malicious JavaScript into pages viewed by other users. Prevent it by escaping/sanitizing all user-controlled output and using Content-Security-Policy headers."
    },
    {
        "question": "what is csrf",
        "answer": "CSRF (Cross-Site Request Forgery) tricks a victim's browser into performing unwanted actions on a site where they are authenticated. Protect with anti-CSRF tokens, SameSite cookies, and checking the Origin/Referer headers."
    },
    {
        "question": "how to hash passwords securely",
        "answer": "Use a modern adaptive hashing algorithm such as bcrypt, argon2, or scrypt. Never store plaintext or simple hashes (MD5/SHA1). Example with bcrypt is shown in the Python section."
    },
    {
        "question": "what is jwt",
        "answer": "JWT (JSON Web Token) is a compact, URL-safe way to represent claims between two parties. Commonly used for stateless authentication. Always verify the signature and check expiration. Store secrets securely."
    },
    {
        "question": "difference between authentication and authorization",
        "answer": "Authentication answers \"Who are you?\" (identity). Authorization answers \"What are you allowed to do?\" (permissions). Always authenticate first, then authorize."
    },
    {
        "question": "explain owasp top 10 briefly",
        "answer": "The OWASP Top 10 lists the most critical web application risks. Key categories include Broken Access Control, Cryptographic Failures, Injection, Insecure Design, Security Misconfiguration, Vulnerable and Outdated Components, Identification and Authentication Failures, Software and Data Integrity Failures, Security Logging and Monitoring Failures, and Server-Side Request Forgery (SSRF)."
    },
]

# ============================================================
# 4. OTHER LANGUAGES & TOOLS
# ============================================================
other_snippets = {
    "javascript reverse string": "function reverseString(str) {\n  return str.split('').reverse().join('');\n}",
    "javascript fetch": '''async function getData(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Request failed");
  return await res.json();
}''',
    "javascript array map": "const numbers = [1, 2, 3, 4];\nconst doubled = numbers.map(n => n * 2);",
    "javascript arrow function": "const add = (a, b) => a + b;\nconst greet = name => `Hello ${name}`;",
    "html basic structure": '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>JagX Page</title>
</head>
<body>
  <h1>Hello from JagX AI</h1>
</body>
</html>''',
    "css center div": '''.center {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}''',
    "css flexbox": '''.container {
  display: flex;
  gap: 16px;
  justify-content: space-between;
  align-items: center;
}''',
    "c++ hello world": '''#include <iostream>
using namespace std;

int main() {
    cout << "Hello from JagX AI" << endl;
    return 0;
}''',
    "c hello world": '''#include <stdio.h>

int main() {
    printf("Hello from JagX AI\\n");
    return 0;
}''',
    "ruby hello world": 'puts "Hello from JagX AI"',
    "git basic commands": "git status\ngit add .\ngit commit -m \"message\"\ngit push origin main\ngit pull\ngit checkout -b feature-branch",
    "docker basic commands": "docker build -t myapp .\ndocker run -p 8000:8000 myapp\ndocker ps\ndocker stop <container_id>\ndocker logs <container_id>",
}

# ============================================================
# 5. MATH & EDUCATION
# ============================================================
math_templates = [
    ("solve {a}x + {b} = {c}", lambda a, b, c: f"{a}x + {b} = {c}\n{a}x = {c - b}\nx = {(c - b) // a if (c - b) % a == 0 else (c - b)/a}"),
    ("solve {a}x - {b} = {c}", lambda a, b, c: f"{a}x - {b} = {c}\n{a}x = {c + b}\nx = {(c + b) // a if (c + b) % a == 0 else (c + b)/a}"),
]

education_qa = [
    {"question": "what is a noun", "answer": "A noun is a word that names a person, place, thing, or idea. Examples: engineer, school, computer, freedom."},
    {"question": "what is a verb", "answer": "A verb is a word that shows action or a state of being. Examples: run, write, is, are, think."},
    {"question": "what is an adjective", "answer": "An adjective describes a noun. Examples: fast, secure, intelligent, reliable."},
    {"question": "area of a circle", "answer": "Area = π × r² where r is the radius."},
    {"question": "area of a rectangle", "answer": "Area = length × width"},
    {"question": "pythagoras theorem", "answer": "In a right-angled triangle: a² + b² = c² where c is the hypotenuse."},
    {"question": "help with homework", "answer": "Of course! Tell me the subject and the exact question, and I will help you step by step."},
]

# ============================================================
# 6. CLEAN CODE / PRINCIPLES
# ============================================================
principles_qa = [
    {
        "question": "clean code tips",
        "answer": "1. Use clear and meaningful names\n2. Keep functions small and focused\n3. Avoid deep nesting\n4. Don't repeat yourself (DRY)\n5. Handle errors properly\n6. Prefer clarity over cleverness"
    },
    {
        "question": "solid principles",
        "answer": "S - Single Responsibility\nO - Open/Closed\nL - Liskov Substitution\nI - Interface Segregation\nD - Dependency Inversion"
    },
    {
        "question": "what is rest api",
        "answer": "REST is an architectural style for networked applications that uses standard HTTP methods (GET, POST, PUT, PATCH, DELETE). It is stateless and resource-oriented."
    },
    {
        "question": "difference between put and patch",
        "answer": "PUT replaces the entire resource. PATCH applies a partial update to a resource."
    },
    {
        "question": "tell me a coding tip",
        "answer": "Write code as if the next person who maintains it is a violent psychopath who knows where you live. Clarity is more important than cleverness."
    },
]

# ============================================================
# GENERATION LOGIC
# ============================================================
def generate_pairs() -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add(q: str, a: str):
        q = q.strip().lower()
        key = q
        if key not in seen and q and a:
            seen.add(key)
            pairs.append({"question": q, "answer": a.strip()})

    # --- Identity ---
    for q in identity_questions:
        answers = identity_answers_map.get(q, IDENTITY_ANSWERS)
        for a in answers:
            add(q, a)
            # slight variations
            add(q + "?", a)
            add(q.capitalize(), a)

    # --- Python ---
    for name, code in python_snippets.items():
        variants = [
            f"python {name}",
            f"python {name} example",
            f"write a python function to {name}",
            f"show me python code for {name}",
            f"how to {name} in python",
            f"python code {name}",
        ]
        for v in variants:
            add(v, code)

    # --- FastAPI ---
    for name, code in fastapi_snippets.items():
        variants = [
            f"fastapi {name}",
            f"fastapi {name} example",
            f"how to {name} in fastapi",
            f"python fastapi {name}",
            f"show fastapi {name}",
        ]
        for v in variants:
            add(v, code)

    # --- Security ---
    for item in security_qa:
        add(item["question"], item["answer"])
        add(item["question"] + "?", item["answer"])
        add("explain " + item["question"], item["answer"])

    # --- Other languages & tools ---
    for name, code in other_snippets.items():
        variants = [
            name,
            f"{name} example",
            f"show me {name}",
            f"how to {name}",
        ]
        for v in variants:
            add(v, code)

    # --- Math generation ---
    for a, b, c in itertools.product(range(2, 9), range(1, 12), range(5, 30)):
        if (c - b) % a == 0:  # keep integer answers mostly
            q = f"solve {a}x + {b} = {c}"
            ans = f"{a}x + {b} = {c}\n{a}x = {c - b}\nx = {(c - b) // a}"
            add(q, ans)
        if (c + b) % a == 0:
            q = f"solve {a}x - {b} = {c}"
            ans = f"{a}x - {b} = {c}\n{a}x = {c + b}\nx = {(c + b) // a}"
            add(q, ans)

    # --- Education & principles ---
    for item in education_qa + principles_qa:
        add(item["question"], item["answer"])
        add(item["question"] + "?", item["answer"])

    # --- Extra identity reinforcement with different phrasings ---
    extra_identity_q = [
        "who is your creator", "who owns you", "which company made you",
        "are you an openai model", "are you based on qwen",
        "do you work for alibaba", "are you related to chatgpt",
    ]
    for q in extra_identity_q:
        add(q, NOT_OTHER if "openai" in q or "qwen" in q or "alibaba" in q or "chatgpt" in q else CREATED_BY)

    return pairs

# ============================================================
# MAIN
# ============================================================
def main():
    print("Generating JagX AI knowledge pairs...")
    pairs = generate_pairs()

    # Shuffle for better distribution
    random.shuffle(pairs)

    # Trim or note if over target
    if len(pairs) > TARGET_COUNT:
        pairs = pairs[:TARGET_COUNT]

    output_path = Path(OUTPUT_FILE)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(pairs)} unique Q&A pairs")
    print(f"Saved to: {output_path.absolute()}")
    print("Identity is strictly preserved (JagX & JRILICENSE only).")

if __name__ == "__main__":
    main()