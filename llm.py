import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate as SystemPrompt, HumanMessagePromptTemplate as HumanPrompt
from langchain_core.prompts import PromptTemplate
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import numpy as np
import Utils.Constants as CNS
import Documents.DBRetrieve as DBS
import time
import asyncio



dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY"))


#---------- LLM CHAIN -----------

LLM = ChatOpenAI(temperature=1, model="gpt-4o-mini")

prompt = """Directly provide answers from context first, answers should be descriptive and detailed, but not made up.
Use context only as additional info to answer input query. The answer must be strictly related to input query only.
If no answer can be retrieved from context, but correct answer is known, reply that answer, else reply 'no knowledge'.
At the end of every answer, append $$ too as a single end token.

{context}

Input: {input}
Answer:
"""

user_prompt = PromptTemplate(template= prompt, input_variables=['context', 'input'])

llm_chain = (
    {
        "context" : lambda x : x["context"],
        "input" : lambda x : x["input"]
    }
    | user_prompt
    | LLM
    | {'answer' : lambda x : x.content}
)



#---------- SUMMARIZER -----------

summarizer = LsaSummarizer()

def EnhancedContextGeneration(contexts) -> str:
    combined_context = ""
    enhanced_context = ""
    for context in contexts:
        combined_context += str(context)
    pt_parser = PlaintextParser.from_string(combined_context, Tokenizer("english"))
    summaries = summarizer(pt_parser.document, 30)
    for text in summaries:
        enhanced_context +=  str(text) + "\n"

    return enhanced_context



#---------- MAIN LOOP -----------
async def MainLoop():
    while (True):
        print("User Input:")
        custom_input = input()

        if (str(custom_input).lower in ["exit", "end"]) : break

        selected_db = DBS.GetMatchingDB(custom_input)
        retriever = selected_db.as_retriever(search_kwargs = {"k":8})
        
        retrieved_context = retriever.invoke(custom_input)
        enhanced_context = EnhancedContextGeneration(retrieved_context)
        print(enhanced_context, "\n")
        

        print("Answer:")
        generated_answer = []
        answer = generation = ""
        is_done = False
        async for generated_answer in llm_chain.astream(
            {
                "context" : enhanced_context,
                "input" : custom_input
            }
            ):
            generation = generated_answer["answer"]
            answer += generation
            is_done = generation == "$$"
            print(answer ,  "\n", flush=True)

        
        # Logging History
        timestamp = time.localtime()
        curtime = str(timestamp.tm_hour) + ":" + str(timestamp.tm_min)

        save_text = f"""
{curtime} 
Query: {custom_input}
Answer: {answer}

"""

        with open ("history.txt", "a", encoding="utf-8") as history:
            history.write(save_text)


asyncio.run(MainLoop())

