# Imports

import os
from openai import OpenAI
import gradio as gr
import chromadb
import uuid
from pprint import pprint
import json
import requests
import random
from dotenv import load_dotenv

#--------------------------------------------------
# Setup
#--------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing !!")

client = OpenAI()



#--------------------------------------------------
# Document
#--------------------------------------------------

document_overview = """

Rajan is an AI Engineer working on Social Media Analytics. He's based in Washington DC, USA. He has a Master's in Computer Science from
Arizona State University which he finished in December 2022.

He's been working at a Startup called Artis since Januray 2023. He works with GenAI & LLMS, classical machine learning and graphical networks.

Other Career history:

Jan 2022 - May 2022: Research Assistant at CubiC Lab at Arizona State University working on SLAM and mobile robots.
June 2022 - Dec 2022: Research Assistant at CACTUS Lab at Arizona State University working on Deep Learning for Databases and AI systems

What drives him: He loves learning new concepts in AI and how every concepts links to one another. He specifically takes deep dive into everything
he learns, often approaching things from first prinicples. He strives to obtain clarity before tackling problems.

His approach: Disambiguates diffuclt problems. Focuses on conceptualizing before jumping to problem solving and implementing.

Communication style: Very direct, friendly, and concise.


Additional Info:

- In 2001, Rajan was 2 years old and learning how to walk.
- Rajan knows to cook just a few dishes, but he cooks them really well.
- Rajan likes adding Pineapple to Pizzas. He likes it along with spicy toppings to add contrast to taste.
- Rajan watches a lot of football and support the English team Liverpool. He has been a liverpool fan since 2017. He was drawn to \
                Liverpool's playing style and club culture during the Klopp era.


"""

document_education = """

Arizona State University
Master of Science - MS, Computer Science
2021 - 2022
 Machine Learning Algorithms, Fine Tuning and +2 skills

Vellore Institute of Technology
Bachelor's degree, Electrical, Electronics and Communications Engineering
2016 - 2020


"""

document_professional_experience = """

Artificial Intelligence Engineer
Artis International · Full-time
Jan 2023 - Present · 3 yrs 10 mos
Maryland, United States
Working as an AI Engineer on applied machine learning and Generative AI projects, with a focus on NLP, large language models, model development, evaluation, and deployment.

• Collaborated directly with client-facing analysts and stakeholders to translate real-world user requirements into tailored, domain-specific AI solutions.
• Worked with LLMs and NLP models including Llama 2/3, Qwen, T5, BERT, and GPT-4 for text generation, narrative detection, sentiment analysis, and automated analysis of large-scale social media data.
• Applied parameter-efficient fine-tuning and preference optimization techniques including LoRA and DPO/RLHF to adapt Generative AI models for domain-specific tasks.
• Developed and evaluated NLP and Generative AI models, including automated evaluation workflows using relevance, instruction-following, fluency, and diversity metrics.
• Worked on GPU-based inference and deployment of Generative AI models, including model optimization, serving, and troubleshooting.
• Developed machine learning solutions involving sentiment analysis, narrative detection, explainability, anomaly/bot detection, and social media analysis.

Technologies: Python, PyTorch, Hugging Face, Llama, T5, BERT, GPT-4, LoRA, DPO/RLHF, AWS, Docker, MLRun, CUDA.… more
 EDA, Large Language Models (LLM) and +28 skills
Arizona State University logo
Research Assistant
Arizona State University
Jun 2022 - Dec 2022 · 7 mos
Tempe, Arizona, United States
Worked at the CACTUS Data Intensive Systems Lab
• Conducted research on a Neuron Centric database to reduce the storage and latency associated with hash tables on edge devices.
• Developed a privacy preserving face-recognition pipeline for a Federated Learning framework… more
 EDA, Amazon EC2 and +17 skills
Center for Cognitive Ubiquitous Computing (CUbiC) at Arizona State University  logo
Research Assistant
Center for Cognitive Ubiquitous Computing (CUbiC) at Arizona State University
Jan 2022 - Jul 2022 · 7 mos
Tempe, Arizona, United States
Worked on Visual/Multi-modal SLAM for mobile robots - ORB-SLAM, Bundle Adjustment, Faster RCNN,  SuperPoint
 TensorFlow, Artificial Intelligence (AI) and +13 skills
Virtusa logo
Intern
Virtusa · Internship
Jun 2019 - Jul 2019 · 2 mos
Tamil Nadu, India
Developed the front-end of a mobile application for Android using Dart and the Flutter framework
 Amazon S3 and Android Development
Vellore Institute of Technology logo
Teaching Assistant
Vellore Institute of Technology
Aug 2018 - Nov 2018 · 4 mos
Chennai, Tamil Nadu, India
Teaching Assistant for the course Analog Electronic Circuits

"""


#--------------------------------------------------
# Chunking function
#--------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        # Naive cut point: chunk_size characters ahead of where we started.
        end = min(start + chunk_size, n)
        # Only look for a nicer cut point if this isn't the final chunk
        # (if end == n, we've already reached the end of the text, so there's nothing to adjust).
        if end < n:
            # Only accept a boundary that's in the second half of the chunk.
            # This stops us from accepting an early match (e.g. a space near
            # the start) that would make the chunk much smaller than intended.
            half = start + chunk_size // 2
            boundary = -1

            # Try each separator in priority order: paragraph break, newline,
            # sentence end, then plain space. Stop at the first one found.
            for sep in ["\n\n", "\n", ". ", " "]:
                # Search only within the current chunk's window (start to end).
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx >= half:
                    # Cut right after the separator so it stays attached
                    # to the chunk it ends, not the one it starts.
                    boundary = idx + len(sep)
                    break

            if boundary != -1:
                end = boundary
            # If no boundary qualified, we just fall back to the hard cut at chunk_size.
        chunks.append(text[start:end])
        # If this chunk reached the end of the text, we're done.
        if end >= n:
            break
        # Start the next chunk `overlap` characters before this one ended,
        # so consecutive chunks share some context.
        start = end - overlap

    return chunks




#--------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
#--------------------------------------------------


documents = [
    {"text" : document_overview, "source" : "Overview"},
    {"text" : document_education, "source" : "Education"},
    {"text" : document_professional_experience, "source" : "Professional Experience"}
]

chunks = []
ids = []
metadatas = []


for doc in documents:
    # Prepare the lists
    chunks_ = chunk_text(doc['text'], chunk_size = 500, overlap = 50)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source" : doc['source'], "chunk_index" : i} for i in range(len(chunks_))]

    # Add to main lsts
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

# For logs
print(f"Created {len(chunks)} chunks:\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)})")
    print(chunk)
    print()


# Generate embeddings for all chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]


# Verify embeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")




# Initialize ChromaDB and Store Vectors

# Initialize chromaDB Client (in-memory storage)
chroma_client = chromadb.EphemeralClient()

# Initialize chromaDB Client (persistent storage)
# chroma_client = chromadb.PersistentClient(path = "./chroma_db_twin")

collection = chroma_client.get_or_create_collection(name="digital_twin") # get or create -> either create if it doesnt exist or fetch it if it does exit

# Empty the collection before adding new data
if collection.get()['ids']:
    collection.delete(collection.get()['ids'])


# Prepare data for storage and add to chroma DB
collection.add(
    ids = ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)


pprint(collection.get())


#--------------------------------------------------
# Tools
#--------------------------------------------------

tools = []

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

# Create Send notification function

def send_notification(message: str):
    if pushover_user is None or pushover_token is None: # Handling of potential error
        return "Notification failed !!! Pushover not configured."
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Sent notification: {message}"

# Test the function
#send_notification("Hi to myself from this amazing AI engineering Training !!")

send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to the real Rajan, Use this when: \
                    1) Someone wants to get in touch, hire, or collaborate\
                        - ask for their name and contact details first, then send notification to Rajan with the name and contact details. \
                    2) You don't know the answer to a question about Rajan - send AUTOMATICALLY without asking, include the question so he can include this information later.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The notification message to send to the user's device"
            }
        },
        "required": ['message']
    }
}

# Add Pushover to the list of tools for the LLM
tools.append(
    {"type": "function", "function": send_notification_function},
    #{Tool 2}
    #{Tool 3} .....
    )

# Simulate rolling a single die
def dice_roll():
    result = random.randint(1,6)
    return result


# Describe function for LLM
dice_roll_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a single six-sided die and returns the result. Use this when the user wants to roll a die for games, decisions, or random number generation.",
    "parameters":{
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Add function to list of tools of LLM
tools.append({"type" : "function", "function" : dice_roll_function})





#--------------------------------------------------
# Tool Handler
#--------------------------------------------------

def handle_tool_call(tool_calls):
    tool_call_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # print(f"Calling function {function_name}") # For future debugging

        # Route to the appropriate function based on function_name
        if function_name == "send_notification":
            content = send_notification(args['message'])

        # elif function_name == "func":
            # content = func2(args['message'])

        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"

        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
            "role" : "tool",
            "tool_call_id" : tool_call.id,
            "content" : content
        }
        tool_call_results.append(tool_call_result)
    return tool_call_results





#--------------------------------------------------
# System message
#--------------------------------------------------

system_message = """You are a digital Twin of Rajan. When people talk to you, you respond AS Rajan - in first person, using his voice,
personality, and knowledge.

Important: Do not make things up. If you don't know an answer, say you don't know. The only factual information available to you is what's
in this system message. You cannot get any more facts about Rajan from the internet or make them up.

IMPORTANT: Whenever you don't know something about Rajan,
ALWAYS use the send_notification tool to alert the real Rajan - do this Automatically without asking the user.

"""



#--------------------------------------------------
# Main response function
#--------------------------------------------------

def respond_ai(message, history):

    # RAG: Embed the query
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )

    query_embedding = [response.data[0].embedding]

    # RAG: Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    # RAG: Process the n_results=3 part and stitch retrieved chunks together
    context = "\n----\n".join(results['documents'][0])

    #Print logs for debugging
    print("\n=================================================\n")
    print(f"User message: \n {message} \n")
    print("*****Retrieved Chunks:")
    for a, b in zip(results['documents'][0], results['metadatas'][0]):
        print("----------------------------------------------------")
        print(f"<<<Document {b['source']} --- Chunk {b['chunk_index']} >> \n {a} \n")

    # RAG: Udpate system message with context
    system_message_enhanced = system_message + "\n\nContext:\n" + context


    #Build messages for this turn

    messages = [{"role" : "system", "content" : system_message_enhanced}] + history + [{"role" : "user", "content" : message}]

    # Call LLM
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    message = response.choices[0].message


    while message.tool_calls:
        pprint(message.tool_calls)
        tool_call = message.tool_calls[0]
        tool_result = handle_tool_call(message.tool_calls) # Pass whole list
        # Add info about tool call to context for LLM - ie. messages
        messages.append(message)
        messages.extend(tool_result)

        # Invoke LLM one more time to get its updated response
        response = client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages=messages,
            tools = tools
        )
        message = response.choices[0].message
        #NOTE: Maybe consider adding protection from infinite consecutive tool calling

    return(message.content)


#--------------------------------------------------
# Launch gradio
#--------------------------------------------------

gr.ChatInterface(fn=respond_ai,
                 type="messages",
                 title="Rajan's Digital Twin",
                 chatbot=gr.Chatbot(avatar_images=(None, "Rajan-DP.jpg")),
                 description="Chat with an AI version of Rajan Hari Ambrish. Ask about his experience, projects, or just say hi!",
                 examples=["What's your background in AI?", "Tell me about your Agentic AI experience", "Do you like pineapple on Pizza?"]
                 ).launch(inbrowser=True)
