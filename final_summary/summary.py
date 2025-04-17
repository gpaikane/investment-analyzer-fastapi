
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


class Summary:

    summary_prompt = """
    Generate a summary using Fundamentals: ```{fundamentals}``` and latest news summary: ```{news_summary}`` of company {ticker}.
    The Summary should not exceed 1000 characters. Also using the above data comment your opinion on long terms investment and short term investment in the company in separate sections, limit the opinion to 500 words.
    and return the generated data to display it nicely in streamlit without code and without any additional attributes.
    
    Note:If news summary is not provided generate summary based on Fundamentals
    """
    chat = ChatOpenAI(temperature=0, model= "gpt-4o-mini")

    @classmethod
    def get_summary(cls, fundamentals, news_summary, ticker):

        summary_prompt = ChatPromptTemplate.from_template(cls.summary_prompt)
        print("NEWSUMMURY----",news_summary)
        message = summary_prompt.format_messages(fundamentals=fundamentals, news_summary=news_summary, ticker=ticker)
        summary = cls.chat.invoke(message)
        return summary