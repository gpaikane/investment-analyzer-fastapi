import os
import  ssl

from celery import Celery
from dotenv import load_dotenv
import logging
from fundamentals.fundamentals import Fundamental
from news.fetchnews import  FetchNews
from  final_summary.summary import Summary

load_dotenv()
""""
REDIS_URL = os.environ.get("REDIS_URL")

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    broker_use_ssl={
        'ssl_cert_reqs': 'CERT_NONE'  # or CERT_OPTIONAL / CERT_REQUIRED
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': 'CERT_NONE'
    },
    result_expires=3600  # auto-expire after 1 hour (3600 seconds)
)

celery_app.conf.broker_transport_options = {"socket_timeout": 500}  # in seconds


celery_app.conf.broker_use_ssl = {
    'ssl_cert_reqs': ssl.CERT_NONE  # <-- use the ssl constant, not a string
}
"""

REDIS_URL = os.environ.get("REDIS_URL")  # Starts with rediss://...

broker_url = REDIS_URL
result_backend = REDIS_URL

broker_use_ssl = {
    'ssl_cert_reqs': None  # 👈 disables strict SSL cert checking (works best with Heroku)
}

result_backend_use_ssl = {
    'ssl_cert_reqs': None
}

celery_app = Celery('yourapp')
celery_app.conf.broker_url = broker_url
celery_app.conf.result_backend = result_backend
celery_app.conf.broker_use_ssl = broker_use_ssl
celery_app.conf.result_backend_use_ssl = result_backend_use_ssl
celery_app.conf.broker_transport_options = {"socket_timeout": 500}  # in seconds



@celery_app.task
def get_fundamental_values( fundamentals: list, ticker_name: str) -> dict:
    fundamental_values = dict()
    try:
        fundamental_values = Fundamental.get_fundamenta_values(fundamentals, ticker_name)
    except Exception as e:
        logging.info(e)

    return fundamental_values


@celery_app.task
def search_news_get_summary( company: str, country_suffix: str) -> dict:
    news= ""
    try:
        news = FetchNews.get_news(company, country_suffix)
    except Exception as e:
        logging.info(e)

    return news

@celery_app.task
def get_final_summary( fundamentals: dict[str,str], news_summary:str, ticker: str) -> dict:
    news = ""
    try:
        news = Summary.get_summary(fundamentals, news_summary, ticker)
    except Exception as e:
        logging.info(e)
    return news