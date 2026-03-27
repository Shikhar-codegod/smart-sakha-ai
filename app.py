import os
import sqlite3
from typing import Dict, List

from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask import session
from openai import OpenAI
from PyPDF2 import PdfReader


load_dotenv()

app = Flask(__name__)

MAX_PDF_CHARS = 12000
ALLOWED_EXTENSIONS = {"pdf"}
CHAT_DB = "chat_history.db"
MAX_CHAT_CONTEXT_MESSAGES = 12


class AIServiceError(Exception):
	pass


def init_db() -> None:
	with sqlite3.connect(CHAT_DB) as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS chat_messages (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
				content TEXT NOT NULL,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)


def save_chat_message(role: str, content: str) -> None:
	with sqlite3.connect(CHAT_DB) as conn:
		conn.execute(
			"INSERT INTO chat_messages (role, content) VALUES (?, ?)",
			(role, content),
		)


def get_chat_history(limit: int = 30) -> List[Dict[str, str]]:
	with sqlite3.connect(CHAT_DB) as conn:
		cursor = conn.execute(
			"""
			SELECT role, content
			FROM chat_messages
			ORDER BY id DESC
			LIMIT ?
			""",
			(limit,),
		)
		rows = cursor.fetchall()

	# Query returns newest first, so reverse for UI and prompt building.
	return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def clear_chat_history() -> None:
	with sqlite3.connect(CHAT_DB) as conn:
		conn.execute("DELETE FROM chat_messages")


def is_allowed_file(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(file_stream, max_chars: int = MAX_PDF_CHARS) -> str:
	try:
		reader = PdfReader(file_stream)
		chunks: List[str] = []
		total_chars = 0

		for page in reader.pages:
			text = page.extract_text() or ""
			if not text:
				continue

			# Stop early once the configured character budget is reached.
			remaining = max_chars - total_chars
			if remaining <= 0:
				break

			clipped = text[:remaining]
			chunks.append(clipped)
			total_chars += len(clipped)

		return "\n".join(chunks).strip()
	except Exception as exc:
		raise ValueError("Could not read the uploaded PDF.") from exc


def call_ai_api(prompt: str, temperature: float = 0.3) -> str:
	api_key = os.getenv("GITHUB_TOKEN")

	if not api_key:
		raise AIServiceError("API key is missing. Set GITHUB_TOKEN in your environment.")

	client = OpenAI(
		base_url="https://models.inference.ai.azure.com",
		api_key=api_key,
	)
	messages = [
		{
			"role": "system",
			"content": "You are a helpful study assistant. Keep responses clear and beginner-friendly.",
		},
		{"role": "user", "content": prompt},
	]

	try:
		response = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=messages,
			temperature=temperature,
			stream=True,
		)
		chunks: List[str] = []
		for chunk in response:
			delta = ""
			if chunk.choices and chunk.choices[0].delta:
				delta = chunk.choices[0].delta.content or ""
			if delta:
				print(delta, end="", flush=True)
				chunks.append(delta)

		if chunks:
			print()

		result = "".join(chunks).strip()
		if not result:
			raise AIServiceError("OpenAI returned an empty response.")
		return result
	except Exception as exc:
		raise AIServiceError("OpenAI API request failed. Please try again.") from exc


def solve_chat_question(question: str) -> str:
	history = get_chat_history(limit=MAX_CHAT_CONTEXT_MESSAGES)
	history_lines = []
	for item in history:
		role_label = "Student" if item["role"] == "user" else "Assistant"
		history_lines.append(f"{role_label}: {item['content']}")

	context_block = "\n".join(history_lines)
	prompt = (
		"You are chatting with a student. Use earlier conversation context when relevant. "
		"If the student asks about a previous question, answer from the chat history provided. "
		"Keep answers clear and beginner-friendly.\n\n"
		f"Previous chat history:\n{context_block or 'No previous messages.'}\n\n"
		f"Current student question: {question}"
	)
	answer = call_ai_api(prompt)
	save_chat_message("user", question)
	save_chat_message("assistant", answer)
	return answer


def summarize_text(text: str, custom_instruction: str = "") -> Dict[str, object]:
	custom_instruction_block = ""
	if custom_instruction:
		custom_instruction_block = (
			"\nAdditional user instruction:\n"
			f"{custom_instruction}\n"
		)

	prompt = (
		"Summarize the following study material for a student.\n"
		"Return plain text only with this structure:\n"
		"Key Points:\n"
		"- point 1\n"
		"- point 2\n"
		"- point 3\n"
		"Explanation:\n"
		"A short, beginner-friendly explanation in 2-4 sentences.\n"
		"Do not return JSON. Do not use code fences.\n"
		f"{custom_instruction_block}\n"
		f"Content:\n{text}"
	)
	ai_raw = call_ai_api(prompt, temperature=0.2)

	cleaned = ai_raw.strip()
	if cleaned.startswith("```"):
		lines = cleaned.splitlines()
		if lines and lines[0].startswith("```"):
			lines = lines[1:]
		if lines and lines[-1].strip() == "```":
			lines = lines[:-1]
		cleaned = "\n".join(lines).strip()

	key_points: List[str] = []
	explanation_lines: List[str] = []
	in_explanation_section = False

	for raw_line in cleaned.splitlines():
		line = raw_line.strip()
		if not line:
			continue

		normalized = line.lower().rstrip(":")
		if "key point" in normalized:
			in_explanation_section = False
			continue
		if normalized in {"explanation", "summary", "short explanation"}:
			in_explanation_section = True
			continue

		bullet = ""
		if line[:1] in {"-", "*", "•"}:
			bullet = line[1:].strip()
		elif len(line) > 2 and line[0].isdigit() and line[1] in {")", "."}:
			bullet = line[2:].strip()

		if bullet:
			key_points.append(bullet)
			continue

		if in_explanation_section:
			explanation_lines.append(line)

	if not key_points:
		fallback_lines = [line.strip("- *•\t") for line in cleaned.splitlines() if line.strip()]
		key_points = [line for line in fallback_lines[:6] if line]

	short_explanation = " ".join(explanation_lines).strip()
	if not short_explanation:
		short_explanation = cleaned

	return {
		"key_points": key_points[:8] or ["No key points could be extracted."],
		"short_explanation": short_explanation,
	}


init_db()


@app.route("/", methods=["GET"])
def index():
	session.clear()
	return render_template("index.html", chat_history=[])


@app.route("/chat", methods=["POST"])
@app.route("/solve", methods=["POST"])
def chat():
	question = (request.form.get("question") or "").strip()
	chat_history = get_chat_history()

	if not question:
		return render_template(
			"index.html",
			doubt_error="Please enter a question.",
			chat_history=chat_history,
		)

	try:
		answer = solve_chat_question(question)
		return render_template(
			"index.html",
			answer=answer,
			chat_history=get_chat_history(),
		)
	except AIServiceError as exc:
		return render_template(
			"index.html",
			question=question,
			doubt_error=str(exc),
			chat_history=chat_history,
		)


@app.route("/clear-chat", methods=["POST"])
def clear_chat():
	clear_chat_history()
	return render_template("index.html", chat_history=[])


@app.route("/pdf", methods=["GET"])
def pdf_page():
	return render_template("pdf.html", custom_instruction="")


@app.route("/summarize", methods=["POST"])
def summarize_pdf():
	uploaded_file = request.files.get("pdf_file")
	custom_instruction = (request.form.get("custom_instruction") or "").strip()

	if not uploaded_file or not uploaded_file.filename:
		return render_template(
			"pdf.html",
			pdf_error="Please upload a PDF file.",
			custom_instruction=custom_instruction,
		)

	if not is_allowed_file(uploaded_file.filename):
		return render_template(
			"pdf.html",
			pdf_error="Invalid file type. Please upload a PDF.",
			custom_instruction=custom_instruction,
		)

	try:
		text = extract_pdf_text(uploaded_file.stream)
		if not text:
			return render_template(
				"pdf.html",
				pdf_error="No readable text found in the PDF.",
			)

		summary = summarize_text(text, custom_instruction=custom_instruction)
		return render_template(
			"pdf.html",
			summary=summary,
			uploaded_name=uploaded_file.filename,
			custom_instruction=custom_instruction,
		)
	except ValueError as exc:
		return render_template(
			"pdf.html",
			pdf_error=str(exc),
			custom_instruction=custom_instruction,
		)
	except AIServiceError as exc:
		return render_template(
			"pdf.html",
			pdf_error=str(exc),
			custom_instruction=custom_instruction,
		)


if __name__ == "__main__":
	app.run(debug=True)