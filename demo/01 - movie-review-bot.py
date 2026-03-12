from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import (BaseModel, Field)
from typing import List
from langsmith import traceable, Client
import os

load_dotenv()

# -- LangSmith Configuration --
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Movie Review Bot Project")
    print(f"LangSmith is configured. - Project: {os.getenv('LANGSMITH_PROJECT')}")

# create movie class
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")
    following_up_questions: List[str] = Field(description="A list of follow-up questions related to the movie")

class SmartMovieReview:
    def __init__(self, model_name: str = "gpt-4o-mini",temperature: float = 0.3):
        # create model
        self.model = init_chat_model(
            model=model_name, 
            temperature=temperature
            ).with_structured_output(MovieReview)
        
        # create prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """
                    You are a movie expert.

                    Guidelines:
                    - Write a short review
                    - Give rating from 0-10
                    - Suggest follow-up questions
                    """),
                ("human", "{title}")
            ]
        )

        self.chain = self.prompt | self.model
    
    @traceable(name="review_movie")
    def review_movie(self, title: str) -> dict:
        try:
            response = self.chain.invoke({"title": title})
            return response.model_dump()
        except Exception as e:
            return {
                "title": title,
                "review":"I'm sorry, I couldn't process the review for this title",
                "rating":0,
                "following_up_questions":[]
            }

@traceable(name="demo_review_bot")
def demo_review_bot():
    bot = SmartMovieReview()

    title = "Spiderman : Coming Home"
    response = bot.review_movie(title)

    print(f"Movie: {response['title']}")
    print(f"Rating: {response['rating']}/10")
    print(f"Review: {response['review']}")
    print(f"Follow-up Questions: {response['following_up_questions']}")


if __name__ == "__main__":
    try:
        demo_review_bot()
    finally:
        Client().flush()