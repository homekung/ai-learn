from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (StrOutputParser, JsonOutputParser, PydanticOutputParser)
from pydantic import (BaseModel, Field)

load_dotenv()

def create_str_output_parser():

    prompt = ChatPromptTemplate.from_template("wire a short poem about {topic}")
    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({"topic" : "nature"})

    print(type(response))
    print(response)


def create_json_output_parser():
    prompt = ChatPromptTemplate.from_template("Return a JSON object with 'name' and 'age' for: {description}")
    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({"description" : "A 38-year-old football player named Messi"})

    print(type(response))
    print(f"Name: {response['name']} Age: {response['age']}")


# create person class
class Person(BaseModel):
    name: str = Field( description="The person's name")
    age: int = Field(description= "The Person's age")
    occupation: str = Field(description="The person's occupation")

def create_pydantic_output_parser():
    
    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=Person)

    # create prompt and correspondent with parser to understand the instruction.
    prompt = ChatPromptTemplate.from_template("Return a JSON object with 'name', 'age' and 'occupation' for: {description}"
                                              ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    response = chain.invoke({"description" : "A 50-year-old artist named Alex"})

    print(type(response))
    print(f"Name: {response.name} Age: {response.age}  Occupation: {response.occupation}")

# create movie class
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")

def create_structured_model():

    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    structured_model = model.with_structured_output(MovieReview)

    result = structured_model.invoke("Review: Inception is a mind-bending thriller. 9/10")
    print(f"Title: {result.title} , {result.review} with Rating {result.rating}")


if __name__ == "__main__":
    #create_str_output_parser()
    #create_json_output_parser()
    #create_pydantic_output_parser()
    create_structured_model()
    




