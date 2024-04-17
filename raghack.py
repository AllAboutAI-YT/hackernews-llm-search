import torch
from sentence_transformers import SentenceTransformer, util
import os
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
import json
import PyPDF2
import re
import requests
from bs4 import BeautifulSoup

load_dotenv()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Anthropic API credentials
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='mistral'
)

def analyse_data(user_input, system_message, vault_embeddings, vault_content, model):
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
        
    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=3000,
        temperature=0.4,
        system=system_message,
        messages=[{"role": "user", "content": user_input_with_context}]
    )
    dataresponse = response.content[0].text
    
    return dataresponse

# Function to get relevant context from the vault based on user input
def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=10):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    # Create a message history including the system message and the user's input with context
    messages = [
        {"role": "system", "content": system_message2},
        {"role": "user", "content": user_input_with_context}
    ]
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model="mistral",
        messages=messages
    )
    # Return the content of the response from the model
    return response.choices[0].message.content

# Function to upload a JSON file and append to vault.txt
def upload_jsonfile(file_path):
    if file_path:
        with open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
            
            # Flatten the JSON data into a single string
            text = json.dumps(data, ensure_ascii=False)
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks
            print(f"JSON file content appended to vault.txt with each chunk on a separate line.")

def scrape_hacker_news(num_pages):
    scraped_posts = []
    
    for page in range(1, num_pages + 1):
        url = f"https://news.ycombinator.com/news?p={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        post_rows = soup.find_all("tr", class_="athing")
        
        for row in post_rows:
            title_element = row.find("span", class_="titleline")
            title = title_element.text.strip()
            link = title_element.find("a")["href"]
            
            post_id = row["id"]
            comments_url = f"https://news.ycombinator.com/item?id={post_id}"
            comments_response = requests.get(comments_url)
            comments_soup = BeautifulSoup(comments_response.content, "html.parser")
            
            comments = comments_soup.find_all("div", class_="comment")
            comment_texts = [comment.find("span", class_="commtext").text.strip() 
                             for comment in comments 
                             if comment.find("span", class_="commtext") is not None]
            
            post_info = {
                "Title": title,
                "Link": link,
                "Comments": comment_texts
            }
            
            scraped_posts.append(post_info)
    
    return scraped_posts

def main():
    # Ask how many pages to scrape
    num_pages = int(input(NEON_GREEN + "Enter the number of pages to scrape from Hacker News: " + RESET_COLOR))
    
    # Scrape the pages using scrape_hacker_news
    scraped_data = scrape_hacker_news(num_pages)
    
    # Save the scraped posts as a JSON file
    with open("scraped_posts.json", "w") as file:
        json.dump(scraped_data, file, indent=4)
    print("Scraped posts saved to scraped_posts.json")
    
    # Upload the scraped JSON file to vault.txt
    upload_jsonfile("scraped_posts.json")
    
    # Load the model and vault content
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
    vault_embeddings = model.encode(vault_content) if vault_content else []
    
    # Convert to tensor and print embeddings
    vault_embeddings_tensor = torch.tensor(vault_embeddings) 
    print("Embeddings for each line in the vault:")
    print(vault_embeddings_tensor)
    
    while True:
        # User input
        user_input = input(YELLOW + "Ask a question about your documents (or type 'exit' to quit): " + RESET_COLOR)
        
        if user_input.lower() == 'exit':
            break
        
        system_message = "You are a helpful assistant that is an expert at extracting the most useful information to the USER's Question"
        response = analyse_data(user_input, system_message, vault_embeddings_tensor, vault_content, model)
        print(NEON_GREEN + "LLM Response: \n\n" + response + RESET_COLOR)

if __name__ == "__main__":
    main()