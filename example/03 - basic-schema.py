from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

def schema_inspection():
    # Step 1: Define prompt template, model and parser using LCEL
    prompt = ChatPromptTemplate.from_template("Tell me the story of: {topic}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    #step 2: Compose with pipe operator (chain)
    chain = prompt | model | parser

    #step 3: Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")

if __name__ == "__main__":
    schema_inspection()