from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class FormalityResponse(BaseModel):
    formality_score: float = Field(description='Formality score of the text')
    reason: str = Field(description='Reasoning behind formality score')


gpt_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0).with_structured_output(FormalityResponse)
SYSTEM_PROMPT = (
    'You will be provided with some text, your task is to determine its formality level on a scale from 0 to 1.'
)

PROMPT_TEMPLATE = (
    'Please analyze the following text and determine its formality level on a scale from 0 to 1, '
    'where 0 is very informal and 1 is very formal.\n\n'
    'Text: "{text}"\n'
)


def gpt_rate_formality(text: str) -> dict[str, float]:
    """utility function that returns formality score rated by llm (gpt-4o-mini)"""
    chat_prompt_template = ChatPromptTemplate(
        [
            ('system', SYSTEM_PROMPT),
            ('human', PROMPT_TEMPLATE),
        ]
    )
    messages = chat_prompt_template.format(text=text)
    formality_response = gpt_llm.invoke(messages)
    formal = formality_response.formality_score
    informal = 1 - formal
    score = {'formal': formal, 'informal': informal}
    return score
