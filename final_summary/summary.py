
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import logging

class Summary:

    summary_prompt = """
    Generate a summary using Fundamentals: ```{fundamentals}``` and latest news summary: ```{news_summary}`` of company {ticker}.
    The Summary should not exceed 1000 characters. Also using the above data comment your opinion on long terms investment and short term investment in the company in separate sections, limit the opinion to 500 words.
    and return the generated data to display it nicely in streamlit without code and without any additional attributes.
    
    Note:If news summary is not provided generate summary based on Fundamentals
    """

    summary_prompt_wt_fundamentals = """
    Generate a summary using Fundamentals: ```{fundamentals}``` of company {ticker}.
    The Summary should not exceed 1000 characters. Also using the above data comment your opinion on long terms investment and short term investment in the company in separate sections, limit the opinion to 500 words.
    and return the generated data to display it nicely in streamlit without code and without any additional attributes.
    """

    summary_prompt_wo_fundamentals = """
    Generate a summary using latest news summary: ```{news_summary}`` of company {ticker}.
    The Summary should not exceed 1000 characters. Also using the above data comment your opinion on long terms investment and short term investment in the company in separate sections, limit the opinion to 500 words.
    and return the generated data to display it nicely in streamlit without code and without any additional attributes.
    """

    chat = ChatOpenAI(temperature=0, model= "gpt-4o-mini")

    @classmethod
    def get_summary(cls, fundamentals, news_summary, ticker):
        summary_prompt= ""
        logging.info("Generating final summary ----")
        logging.info(fundamentals, news_summary, ticker)
        if (fundamentals is not None and  len(fundamentals) > 0) and (news_summary is None or len(news_summary) == 0):
            logging.info("fundamentals are not generated, however news_summary is generated")
            summary_prompt = ChatPromptTemplate.from_template(cls.summary_prompt_wt_fundamentals)

        elif (fundamentals is  None or  len(fundamentals) == 0) and (news_summary is not None and len(news_summary) > 0):
            logging.info("fundamentals are generated, however news_summary is not generated")
            summary_prompt = ChatPromptTemplate.from_template(cls.summary_prompt_wo_fundamentals)
        else:
            logging.info("fundamentals are generated, however news_summary is generated")
            summary_prompt = ChatPromptTemplate.from_template(cls.summary_prompt)

        logging.info("NEWSUMMURY----",news_summary)
        if(summary_prompt == ""):
            return ""
        message = summary_prompt.format_messages(fundamentals=fundamentals, news_summary=news_summary, ticker=ticker)
        summary = cls.chat.invoke(message)
        return summary.content