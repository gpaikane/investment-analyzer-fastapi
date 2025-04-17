from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict
import create_chroma_db

from fundamentals.fundamentals import Fundamental
from news.fetchnews import FetchNews
from forecast.forecast import ForeCast
from final_summary.summary import Summary
from fastapi import FastAPI, Query
from pydantic import BaseModel


class TextInput(BaseModel):
    text: str

class IntInput(BaseModel):
    fundamental_count: int

class ListInput(BaseModel):
    text: list

class DictMultiInput(BaseModel):
    fundamentals_values: Dict[str, str]
    new_summary: str
    ticker: str

app = FastAPI()

@app.get("/get_ticker/")
async def get_ticker(input_data: TextInput):
    """
    Get ticker supported by yfinance by calling llm
    :param input_data: Text/Info on company provided by user
    :return:ticker
    """
    user_provided_text = input_data.text
    ticker = Fundamental.get_company_yfinance_ticker(user_provided_text)
    return ticker

@app.get("/get_top_fundamentals/")
async def get_top_fundamentals(input_data: IntInput):
    """
    Get top fundamentals which investors use from llm
    :param input_data: count of fundamentals
    :return: list of top n fundamentals, `n` is specified in the input data
    """
    user_provided_text = input_data.fundamental_count
    fundamentals = Fundamental.get_top_n_fundamentals(int(user_provided_text))
    return fundamentals


@app.get("/get_fundamentals_values/")
async def get_fundamentals_values(fundamentals: List[str]= Query(...), ticker: str= Query(...)):
    """
    Calculate values for each fundamental
    :param fundamentals: List
    :param ticker
    :return: dictionary of fundamentals and its values
    """
    fundamental_values = await Fundamental.get_fundamenta_values(fundamentals,ticker )
    return fundamental_values

@app.get("/get_company_details/")
async def get_company_details(text: str):
    """
    Get company name and suffix for the country where the company is located, for indian companies the suffix is `in`
    :param text: Text/Info on company provided by user with the country where the company is listed
    :return:  Company name and country suffix
    """
    company, suffix = FetchNews.get_company_country_details(text)
    return company,suffix

@app.get("/get_news/")
async def get_news(company: str, suffix: str):
    """
    Retrive top 5 latest news for the company and summarize it
    :param company:  company name
    :param suffix:  countru suffix
    :return: summary of news
    """
    news = FetchNews.get_news(company, suffix)
    return news

@app.get("/get_summary/")
async def get_summary(inputData: DictMultiInput):
    """
    Get detailed summary of how the short and long terms investments outlooks is for the company
    :param inputData:  Input data contains fundamentals_values, news summary and company ticker
    :return: Markdown formated summary
    """
    summary = Summary.get_summary(inputData.fundamentals_values, inputData.new_summary, inputData.ticker)
    return summary

@app.get("/get_forecasted_data/")
async def get_forecasted_data(ticker:str):
    """
    Get 60 days forecat of the company
    :param ticker: Ticker of the company supported by yfinance
    :return: Forecasted data
    """
    return ForeCast.get_forecasted_data(ticker)