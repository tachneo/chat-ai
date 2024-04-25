import sqlite3
import tkinter as tk
from tkinter import simpledialog, font as tkfont
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sympy import sympify, symbols, solve, S
import os
from pint import UnitRegistry
import scipy.constants as const
from chempy import balance_stoichiometry
from sympy import symbols, Eq, solve
import spacy

# Load the spaCy English language model
nlp = spacy.load('en_core_web_sm')

ureg = UnitRegistry()
Q_ = ureg.Quantity

# Load spaCy model for supplementary NLP tasks
nlp = spacy.load('en_core_web_sm')

db_path = 'chatbot.db'  # Define your database path here

def setup_database():
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_answer(question):
    # Normalizing the question
    question = question.lower().strip()

    # Open database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM qa_pairs')
    qa_pairs = cursor.fetchall()
    conn.close()

    # Check if database has data
    if not qa_pairs:
        return "I'm still learning. Can you teach me?", None

    questions, answers = zip(*qa_pairs)
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    # Transform the query to the same vector space as the corpus
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, question_vectors)

    # Find the highest similarity score
    max_similarity_index = similarities.argmax()
    max_similarity = similarities[0, max_similarity_index]

    # Consider answer if similarity is above a threshold, e.g., 0.5
    if max_similarity > 0.5:
        return answers[max_similarity_index], questions[max_similarity_index]
    
    # If no similar question found, generate response using GPT-2
    return generate_response(question), None
    


def add_or_update_qa_pair(question, answer):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT answer FROM qa_pairs WHERE question = ?', (question,))
    existing_answer = cursor.fetchone()

    if existing_answer:
        cursor.execute('UPDATE qa_pairs SET answer = ? WHERE question = ?', (answer, question))
    else:
        cursor.execute('INSERT INTO qa_pairs (question, answer) VALUES (?, ?)', (question, answer))

    conn.commit()
    conn.close()

def solve_math_problem(question, decimal_places=2):
    try:
        # Strip off 'solve' keyword and any non-math characters that might confuse the parser
        clean_question = re.sub(r'[^0-9\+\-\*\/\^\(\)\=\.]', ' ', question.replace('solve', '', 1).strip())
        
        # Add explicit multiplication where necessary
        clean_question = add_explicit_multiplication(clean_question)
        
        # Use sympify to parse the cleaned question
        expr = sympify(clean_question)
        
        # Identify variables in the expression
        variables = expr.free_symbols
        if not variables:
            # Evaluate directly if no variables and format the result
            evaluated_result = expr.evalf()
            formatted_result = f"{evaluated_result:.{decimal_places}f}"  # Format to specified decimal places
            return f"The solution is {formatted_result}"
        
        # If there are variables, attempt to solve the expression
        solution = solve(expr, *variables)
        
        # Format the output depending on the type of solution
        if isinstance(solution, list):
            return ', '.join([f"{str(var)} = {float(sol):.{decimal_places}f}" for var, sol in zip(variables, solution)])
        elif isinstance(solution, dict):
            return ', '.join([f"{str(var)} = {float(sol):.{decimal_places}f}" for var, sol in solution.items()])
        return f"The solution is {solution}"
    except Exception as e:
        return f"Could not solve the problem: {str(e)}"
        
class QnAGuiApp:
    def __init__(self, root):
        self.root = root
        self.conversation_history = []
        root.title("Tachy Chat")
        root.configure(bg='#9604c7')

        chat_font = tkfont.Font(family="Helvetica", size=12)
        chat_frame = tk.Frame(root, bg='white', bd=2)
        chat_frame.pack(padx=20, pady=20, fill='both', expand=True)

        self.chat_display = tk.Text(chat_frame, height=15, width=50, bg='white', state='disabled', font=chat_font)
        self.chat_display.pack(side='left', fill='both', expand=True)

        chat_scrollbar = tk.Scrollbar(chat_frame, command=self.chat_display.yview)
        chat_scrollbar.pack(side='right', fill='y')
        self.chat_display['yscrollcommand'] = chat_scrollbar.set

        entry_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.question_entry = tk.Entry(root, width=40, font=entry_font, bg='white', fg='#9604c7')
        self.question_entry.pack(pady=10)
        self.question_entry.bind("<Return>", self.search_question)

        send_button = tk.Button(root, text="Send", command=self.search_question, bg='white', fg='#9604c7', font=entry_font, relief='groove')
        send_button.pack()

    def categorize_and_respond(question):
        if re.search(r'\b(solve|calculate|how much is|what is)\b', question):
            return solve_math_problem(question)
        elif re.search(r'\b(where do you live|your name|who built you)\b', question):
            return handle_personal_questions(question)
        else:
            return generate_relevant_response(question)
    
    def handle_personal_questions(question):
        responses = {
            "where do you live": "I exist in the cloud, ready to assist you from anywhere!",
            "your name": "I'm Tachy, your friendly virtual assistant.",
            "who built you": "I was developed by Rohit Singh, an innovative software developer."
        }
        return responses.get(question.lower(), "I'm not sure how to answer that. Can you rephrase?")

    def generate_dynamic_response(query):
        # Assume `chat_model` is a pre-loaded large language model like GPT-3
        response = chat_model.generate(query, max_length=50)
        return response
        
    def generate_relevant_response(question):
        # Assume generate_response is a method that uses GPT-2 or similar
        response = generate_response(question).split(".")[0]  # Take the first sentence only
        if not response.endswith("."):
            response += "."
        return response
        
    def adjust_tone_based_on_input(question, response):
        if question.isupper():  # User is shouting
            return response.upper()  # Respond in kind
        return response
    
    # Example of handling predefined questions
    def handle_predefined_questions(question):
        responses = {
            "where do you live": "I am digital and exist on servers!",
            "who built you": "I was developed by Rohit Singh.",
            "how are you": "I am good, thanks for asking. How can I assist you today?",
            "what is your name": "My name is Tachy, your friendly AI assistant."
        }
        # Normalize question for better matching
        question = question.lower().strip()
        return responses.get(question, None)
    
    def search_question(self, event=None):
        question = self.question_entry.get().strip()
        if not question:
            return
    
        # Enhanced detection for math problems using a regular expression
        math_pattern = re.compile(r'(?<!\w)(solve|calculate|what is)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?([+\-*/^]\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)*)(?!\w)')
        if math_pattern.search(question.lower()):
            try:
                answer = solve_math_problem(question)
                self.update_chat(f"You: {question}")
                if answer:
                    self.update_chat(f"Bot: {answer}")
                else:
                    self.update_chat("Bot: I'm not sure how to solve this. Can you simplify the question?")
            except Exception as e:
                self.update_chat(f"Bot: Error solving math problem: {str(e)}")
        else:
            # Use advanced natural language processing for better question understanding
            answer, matched_question = get_answer(question)
            self.update_chat(f"You: {question}")
            if answer:
                self.update_chat(f"Bot: {answer}", matched_question)
            else:
                # If no answer is found, engage the user for learning
                self.update_chat("Bot: I don't know the answer. Can you tell me?")
                self.get_user_input(question)
    
        self.question_entry.delete(0, 'end')

    def get_user_input(self, question):
        answer = simpledialog.askstring("Input", "What should be the answer?", parent=self.root)
        if answer:
            add_or_update_qa_pair(question, answer)
            self.update_chat("Bot: Thank you! I've learned something new.")
            self.add_edit_button(question)

    def add_edit_button(self, question):
        edit_button = tk.Button(self.root, text="Edit", command=lambda: self.edit_answer(question), bg='white', fg='#9604c7')
        self.chat_display.window_create(tk.END, window=edit_button)
        self.chat_display.insert(tk.END, '\n\n')
    
    def update_chat(self, message, matched_question=None):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + '\n')
        if matched_question:
            self.add_edit_button(matched_question)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def edit_answer(self, question):
        current_answer = simpledialog.askstring("Edit Answer", "Enter the new answer:", parent=self.root)
        if current_answer:
            add_or_update_qa_pair(question, current_answer)
            self.update_chat(f"Bot (Updated): {current_answer}")
            
    def extract_variables(expr):
        # Regex to find all potential variables (letters or words that could act as symbolic variables)
        potential_vars = re.findall(r'[a-zA-Z_]+', expr)
        # Filter out known function names in SymPy
        known_functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'E']
        variables = set(potential_vars) - set(known_functions)
        return list(variables)

    previous_question = None  # Keep track of the last question asked
    
    def handle_repetition(question):
        global previous_question
        if question == previous_question:
            return "As I mentioned earlier, " + generate_response(question)
        else:
            previous_question = question
            return generate_response(question)

def add_explicit_multiplication(expr):
    # Regex to insert * between digit and parenthesis or between parentheses without operator
    expr = re.sub(r'(\d)(\()', r'\1*\2', expr)
    expr = re.sub(r'(\))(\d)', r'\1*\2', expr)
    expr = re.sub(r'(\))(\()', r'\1*\2', expr)
    return expr

def evaluate_with_units(expression):
    try:
        # Parse the expression with units
        result = Q_(expression).to_base_units().magnitude
        return f"The result is {result:.2f} in base units"
    except Exception as e:
        return f"Error evaluating expression with units: {str(e)}"

def balance_chemical_equation(reactants, products):
    try:
        reac, prod = balance_stoichiometry({reactants}, {products})
        balanced_eq = ' + '.join(f"{v} {k}" for k, v in reac.items()) + ' -> ' + ' + '.join(f"{v} {k}" for k, v in prod.items())
        return f"Balanced equation: {balanced_eq}"
    except Exception as e:
        return f"Could not balance equation: {str(e)}"

def calculate_kinetic_energy(mass, velocity):
    try:
        ke = 0.5 * mass * (velocity ** 2)
        return f"The kinetic energy of a {mass} kg object moving at {velocity} meters per second is {ke} joules."
    except Exception as e:
        return f"Error calculating kinetic energy: {str(e)}"


def handle_question(question):
    # Use regex to identify numbers, words, and basic mathematical operations
    numbers = map(float, re.findall(r"\d+\.?\d*", question))
    words = re.findall(r"[a-zA-Z]+", question)

    # Natural language handling to categorize the question
    if 'kinetic energy' in question:
        try:
            mass, velocity = numbers
            return calculate_kinetic_energy(mass, velocity)
        except Exception as e:
            return f"Error in calculating kinetic energy: {str(e)}"

    elif 'solve' in question and 'equation' in question:
        # For algebraic solutions, the equation should be clearly specified
        try:
            equation = ' '.join(words)
            return solve_algebraic_equation(equation)
        except Exception as e:
            return f"Failed to solve the equation: {str(e)}"

    elif 'balance' in question and 'equation' in question:
        # Extracting chemical equation components for balancing
        try:
            reactants, products = question.split('->')
            return balance_chemical_equation(reactants, products)
        except Exception as e:
            return f"Error balancing the equation: {str(e)}"

    elif 'molar mass' in question:
        try:
            compound = ' '.join(words)
            return calculate_molar_mass(compound)
        except Exception as e:
            return f"Error calculating molar mass: {str(e)}"

    else:
        # This could be further enhanced by integrating with a large language model for dynamic response generation
        return generate_dynamic_response(question)

def generate_dynamic_response(question):
    # Analyze the question
    doc = nlp(question)
    nouns = [chunk.text for chunk in doc.noun_chunks]  # Extract noun phrases for context
    verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']  # Extract verbs for action context

    # Check if any entities or keywords are recognized
    if nouns:
        response = f"That's interesting, tell me more about {nouns[0]}."
    elif verbs:
        response = f"Can you explain how you {verbs[0]} that?"
    else:
        response = "I'm still learning about that topic. Can you ask something else?"

    return response

def generate_response(question):
    # Analyze the question
    doc = nlp(question)
    nouns = [chunk.text for chunk in doc.noun_chunks]  # Extract noun phrases for context
    verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']  # Extract verbs for action context

    # Check if any entities or keywords are recognized
    if nouns:
        response = f"That's interesting, tell me more about {nouns[0]}."
    elif verbs:
        response = f"Can you explain how you {verbs[0]} that?"
    else:
        response = "I'm still learning about that topic. Can you ask something else?"

    return response

def calculate_kinetic_energy(mass, velocity):
    # Kinetic energy = 0.5 * mass * velocity^2
    ke = 0.5 * mass * (velocity ** 2)
    return f"The kinetic energy is {ke} joules"

def balance_chemical_equation(reactants, products):
    reac, prod = balance_stoichiometry({reactants}, {products})
    balanced_eq = ' + '.join(f"{v} {k}" for k, v in reac.items()) + ' -> ' + ' + '.join(f"{v} {k}" for k, v in prod.items())
    return f"Balanced equation: {balanced_eq}"


def solve_algebraic_equation(equation):
    try:
        x = symbols('x')
        eq = eval("Eq(" + equation.replace("=", ",") + ")")
        solution = solve(eq, x)
        if solution:
            return f"The solution to the equation {equation} is x = {solution[0]}"
        else:
            return "The equation has no solution."
    except Exception as e:
        return f"Error solving the equation: {str(e)}"

# Run the application
if __name__ == "__main__":
    setup_database()
    root = tk.Tk()
    root.iconbitmap(r'C:\Users\ASUS\OneDrive\Desktop\Project\Tachy\tachy.ico')  # Set the window icon
    app = QnAGuiApp(root)
    root.mainloop()
