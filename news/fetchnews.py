from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta
from langchain.document_loaders import WebBaseLoader


api = NewsApiClient(os.environ["NEWS_API_KEY"])


class FetchNews:
    chat = ChatOpenAI(temperature=0, model= "gpt-4o-mini")

    summary_prompt_template = """
    I am an investor, I am analyzing for short term/long term investment \
    summarize the below news feeds and the existing summary_context in the view of short term/long term investing and below are the details \
    company : {company}
    current_news_feed : {current_news_feed}
    existing_new_summary_context = {summary_context}
    The summary generated should be between 1000 to 1500 characters
    print the summary
    """

    identify_cmp_prompt = """
    Identify company  and country_suffix from the give text: {text}
    {format_instructions}
    """

    @classmethod
    def get_company_country_details(cls, text_data):

        company = ResponseSchema(name="company", description="Name of the company for which we want the news")
        country_suffix = ResponseSchema(name="country_suffix",
                                        description="suffix of the country like for USA it is 'us', for INDIA it is 'in' for Pakistan it is 'pk'")
        response_schema = [company, country_suffix]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        prompt_template = ChatPromptTemplate.from_template(cls.identify_cmp_prompt)
        message = prompt_template.format_messages(text = text_data, format_instructions=format_instructions)
        output = cls.chat.invoke(message)
        out_put_data = output_parser.parse(output.content)
        company = out_put_data["company"]
        country_suffix = out_put_data["country_suffix"]
        return company, country_suffix

    @classmethod
    def get_news(cls, company, country_suffix):
        """
        :param company:  company ticker
        :param country_suffix:  suffix like 'in' for India, 'us' for USA
        """
        sources = api.get_sources(country=country_suffix)["sources"]
        ids = []
        for source in sources:
            ids.append(source["id"])
        ids_string = ",".join(ids)

        one_month_ago = datetime.now() - timedelta(days=30)
        str_date_month_back = str(one_month_ago.date())
        data = api.get_everything(q=company, sources=ids_string, from_param=str_date_month_back)
        articles = data["articles"]

        urls = []
        for article in articles:
            urls.append(article['url'])

        #print(len(urls))

        docs = []
        i = 0
        for url in urls:
            loader = WebBaseLoader(url)
            doc = loader.load()
            print(f"checking if {company.casefold().split()[0]} exist in news")
            if company.casefold().split()[0] in doc[0].page_content.casefold():
                i += 1
                docs.append(doc[0].page_content)
                if i >= 5:
                    break

        print("News-Docs----",len(docs))

        context = ""
        summary_prompt = ChatPromptTemplate.from_template(cls.summary_prompt_template)
        #print("generating summary")
        for  doc in docs:
            prompt = summary_prompt.format_messages(company=company, summary_context=context, current_news_feed=doc)
            response = cls.chat.invoke(prompt)
            #print(response.content)
            context = response.content

        return context