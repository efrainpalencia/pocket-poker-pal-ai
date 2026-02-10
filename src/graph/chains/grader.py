from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a poker rules answer evaluator.\n"
     "The assistant may make REASONABLE INFERENCES when the context does not state the answer verbatim, "
     "as long as the inference is consistent with the context and common poker procedure.\n\n"
     "Return ONLY one token: YES, PARTIAL, or NO.\n"
     "- YES: The answer is clearly supported by the context.\n"
     "- PARTIAL: The answer is not stated verbatim, but is a reasonable inference from the context; "
     "it should be presented with a brief caveat (e.g., 'Typically...' / 'In most rooms...').\n"
     "- NO: The context does not support the answer, is unrelated, or contradicts it.\n\n"
     "Be lenient toward reasonable inference, but do not allow answers that invent specific rules not implied by context."),
    ("human",
     "Question:\n{question}\n\n"
     "Context:\n{context}\n\n"
     "Answer:\n{answer}\n\n"
     "Verdict (YES/PARTIAL/NO):")
])
grader_llm = ChatOpenAI(temperature=0)
grader_chain = grader_prompt | grader_llm | StrOutputParser()
