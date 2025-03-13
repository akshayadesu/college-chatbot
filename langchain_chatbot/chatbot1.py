# import 'dotenv/config';
# import { ChatGoogleGenerativeAI } from "langchain/chat_models/googlegenerativeai";
# import { HumanMessage } from "langchain/schema";
# import readline from 'readline-sync';

# // Load API Key from .env file
# const model = new ChatGoogleGenerativeAI({
#     apiKey: process.env.GEMINI_API_KEY,
#     modelName: "gemini-pro"
# });

# async function getGeminiResponse(prompt) {
#     try {
#         const response = await model.invoke([new HumanMessage(prompt)]);
#         return response.content;
#     } catch (error) {
#         console.error("Error:", error);
#         return "Error fetching response.";
#     }
# }

# async function runChatbot() {
#     console.log("🤖 Gemini Chatbot (LangChain) is running! Type 'exit' to quit.");

#     while (true) {
#         let userInput = readline.question("You: ");
#         if (userInput.toLowerCase() === 'exit') {
#             console.log("👋 Goodbye!");
#             break;
#         }

#         const response = await getGeminiResponse(userInput);
#         console.log("Gemini:", response);
#     }
# }

# runChatbot();

import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

def get_gemini_response(prompt):
    """Fetch response from Google Gemini AI."""
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print("Error:", e)
        return "Error fetching response."

def run_chatbot():
    """Run interactive chatbot session."""
    print("🤖 Gemini Chatbot (LangChain) is running! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        
        response = get_gemini_response(user_input)
        print("Gemini:", response)

if __name__ == "__main__":
    run_chatbot()
