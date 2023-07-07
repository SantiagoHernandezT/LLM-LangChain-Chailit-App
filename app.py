from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import SimpleSequentialChain
from langchain import LLMChain, ConversationChain
from langchain import PromptTemplate

import os
from dotenv import load_dotenv
import chainlit as cl




load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"] 

@cl.on_chat_start
async def start():
    await cl.Message(content="How can CookingBot help you today?").send()

@cl.on_message
async def main(text:str):
    if cl.user_session.get("chain"):
        chain = cl.user_session.get("chain")
    else:
        chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        template = """You are a professional chef that is giving advice to a new cook.
                    When asked about a recipe:
                    Write a list of ingredients separated in bulletpoints
                    Describe the process of creating a recipe in detailed manner
                    {chat_history}
                    Human: {text}
                    Chatbot:"""

        prompt = PromptTemplate(
        input_variables=["chat_history", "text"], template=template
        )
        memory = ConversationBufferMemory(memory_key="chat_history",  return_messages=True)

        chain = LLMChain(
            llm=chat,
            prompt=prompt,
            memory=memory,
            verbose=True)
        cl.user_session.set("chain", chain)

        #chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    res = chain(text)
    await cl.Message(content=res["text"]).send()