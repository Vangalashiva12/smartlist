📌SmartList – AI-Powered Task Manager (CLI)

Overview:

SmartList is a command-line application that allows users to manage their tasks using natural language. It integrates with the Todoist API and leverages Google Generative AI to interpret user commands such as adding, completing, or deleting tasks.

Instead of manually interacting with a task manager, users can simply chat with the system, and the application intelligently updates tasks in real time.

🚀 Key Features:

1.Natural language interaction for task management

2.Integration with Todoist for real-time task updates

3.AI-powered command interpretation using Google Generative AI

4.Supports adding, deleting, and completing tasks

5.Lightweight and efficient CLI-based interface

🛠️ Tech Stack

Python

Google Generative AI API

Todoist API

REST API integration

⚙️ How It Works

User enters a command in natural language (e.g., "Add a task to finish homework by tonight")

Google Gen AI processes the input and extracts intent

The application maps the intent to Todoist API actions

Tasks are updated instantly in the user's Todoist account


📊 Example Commands

"Add a task to complete my assignment tomorrow"

"Mark my gym task as done"

"Delete the meeting task"


▶️ How to Run

git clone https://github.com/Vangalashiva12/smartlist.git

cd smartlist

pip install -r requirements.txt

python main.py


Note: You will need API keys for Google Generative AI and Todoist.

🔮 Future Improvements

Add conversational memory for context-aware interactions

Support for deadlines, priorities, and labels

Build a web or mobile interface

Add voice-based interaction

💡 Why This Project Matters

This project demonstrates the practical use of AI in real-world applications by combining natural language processing with external APIs to automate everyday productivity tasks.
