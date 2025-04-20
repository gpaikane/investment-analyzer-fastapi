import logging
from typing import Optional, Any

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool,AgentType
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from  create_pinecone_db import vectorstore



class Fundamental:
    # Prompt templates
    find_ticker_value = """
    using the input_value identify the ticker which yfinance library uses here are some examples:
    MSFT or microsoft the ticket is MSFT.
    For TCS a indian company the ticker is TCS.NS or TCS.BO (NS for NSE, BO for BSE), we can use TCS.NS as default unless specified,
    similarly for other stock exchanges.

    Return only the ticker like MSFT, TCS.NS , TCS.BO etc.
    The ticket must be supported by yfinance or yahoo finance
    input_value is given here: {input_value}

     {format_instructions}
    
    """

    promt_find_fundamentals = """
    Generate a list of {n} top market fundamentals which are used to access stock before investing.
    We should be able to calculate or get the fundamentals using yfinance python library.
    {format_instructions}
    
    """

    get_function_prompt = """
    use a python code to print() dir(yf.Ticker)
    make you must print the final result
    
    {format_instructions}
    
    VERY IMPORTANT NOTES:
    Note 1: use the provided tool to execute python commands. 
    Note 2: Always print(...) output at the end in the generated python code

    """

    get_important_methods = """
    Here are the important attributes and methods {methods} of yfinance.Ticker among them Identify the methods which we can use to calculate {fundamental}.
    {format_instructions}
    """

    calculate_fundamental_template = """
    
    calculate and print with `print(...)` {fundamental} value for {ticker_name} ticker, here are some yfinance python library methods suggested {methods},  \
    and here is the context of the methods {combined_context}  to be referred \
    in case the {methods}  provided are not useful, you can use a new method from yfinance python library. \
    generate python code for identifying  or calculating given {fundamental} and printing the output \
    make sure you print the output using `print(...)`
    
    VERY IMPORTANT NOTES:
    Note 1: use the provided tool to execute python commands. 
    Note 2: In case you are unable to find an answer try with a new approach using yfinance python library
    Note 3: Always print(...) output at the end in the generated python code

    """
    python_repl = PythonREPL()

    # You can create the tool to pass to an agent
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command.  you must print output with `print(...)`.",
        func=python_repl.run,
    )

    chat = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini" )

    agent = create_python_agent(
        chat,
        tool=repl_tool,
        verbose=False,
        handle_parsing_errors=True,
        agent=AgentType.SELF_ASK_WITH_SEARCH

    )

    embedding = OpenAIEmbeddings()


    @classmethod
    def invoke_agent(cls, message_to_agent, retry_num) -> Optional[dict[str, Any]]:

        message_funcs = None
        retries = 0
        while retries < retry_num:
            try:
                message_funcs = cls.agent.invoke(message_to_agent)
                logging.info("Agent Message:.......", message_funcs)
                return  message_funcs
            except Exception as e:
                logging.exception(e)
                logging.info("Triggering agent once again....")
                retries += 1

        return  message_funcs

    @classmethod
    def get_context_from_methods(cls, methods: list) -> str:

        selected_methods_context = []
        for method in methods:
            print(method)
            value = vectorstore.max_marginal_relevance_search(method, k=1)
            print("VALUE:----", value)
            selected_methods_context.append(value[0].page_content)
        combined_context = "\n------\n".join(selected_methods_context)
        return combined_context

    OpenAIEmbeddings
    @classmethod
    def get_yfianance_function_list(cls):
        logging.info("getting methods supported by yfinance..........")
        methods = ResponseSchema(name="methods",
                                 description="methods supported by yfinance python librabry which will be used to identify the fundamental later",
                                 type="list")
        response_schema = [methods]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        promt_format_instructions_template = ChatPromptTemplate.from_template(cls.get_function_prompt)
        get_function_message = promt_format_instructions_template.format_messages(
            format_instructions=format_instructions)
        logging.info("trying to get_yfianance_function_list")
        message_funcs = cls.invoke_agent(get_function_message, 3)
        return message_funcs

    @classmethod
    def get_company_yfinance_ticker(cls, text: str) -> str:

        ticker = ResponseSchema(name="output_value",
                                description="The ticker name supported by yfianance python librabry")
        response_schema = [ticker]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        promt_input_value_template = ChatPromptTemplate.from_template(cls.find_ticker_value)
        message = promt_input_value_template.format_messages(input_value=text, format_instructions=format_instructions)
        response = cls.chat.invoke(message)
        output = output_parser.parse(response.content)

        return output["output_value"]

    @classmethod
    def get_top_n_fundamentals(cls, count: int) -> list:
        """
        :param count: the number of desired fundamentals
        :return: list of fundamentals
        """
        fundametal_list = ResponseSchema(name="fundamentals",
                                         description="list of fundamentals seperated by comma in list format ",
                                         type="list")
        response_schema = [fundametal_list]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        promt_find_fundamentals_template = ChatPromptTemplate.from_template(cls.promt_find_fundamentals)
        message = promt_find_fundamentals_template.format_messages(n=count, format_instructions=format_instructions)
        response = cls.chat.invoke(message)
        output = output_parser.parse(response.content)
        return output["fundamentals"]

    @classmethod
    def get_fundamenta_values(cls, fundamentals: list, ticker_name: str) -> dict:
        """

        :param fundamentals:  list of fundamentals to be calculated
        :param ticker_name:  ticker name as per yfianance
        :return:  dictionary of fundamentals
        """

        fundamentals_values = dict()

        y_finance_methods = cls.get_yfianance_function_list()

        def select_sugested_methods(fundamental, methods_list) -> list:

            methods = ResponseSchema(name="methods",
                                     description="methods or methods supported by yfianance python librabry which can be used to identify or calculate the fundamental value, include max 3 methods",
                                     type="list")
            response_schema = [methods]
            output_parser = StructuredOutputParser.from_response_schemas(response_schema)
            format_instructions = output_parser.get_format_instructions()
            get_important_methods_prompt = ChatPromptTemplate.from_template(cls.get_important_methods)
            message = get_important_methods_prompt.format_messages(methods=methods_list,
                                                                   format_instructions=format_instructions,
                                                                   fundamental=fundamental)
            response = cls.chat.invoke(message)
            output = output_parser.parse(response.content)
            return output["methods"]

        def calculate_fundamental_value(fundamental, selected_methods, context, ticker_name):

            calculate_fundamental_prompt = ChatPromptTemplate.from_template(cls.calculate_fundamental_template)
            message = calculate_fundamental_prompt.format_messages(ticker_name=ticker_name, methods=selected_methods,
                                                                   fundamental=fundamental, combined_context=context)
            fundamental_value = cls.invoke_agent(message,3)
            return fundamental_value


        for fundamental in fundamentals:
            selected_methods = select_sugested_methods(fundamental, y_finance_methods)
            context = cls.get_context_from_methods(selected_methods)
            value = calculate_fundamental_value(fundamental, selected_methods, context, ticker_name)
            selected_value = value['output'] if type(value) == dict else value
            fundamentals_values[fundamental] = selected_value if selected_value not in "I don't know." else None

            # Retry if the llm agent returns I don't know
            if fundamentals_values[fundamental] is None:
                value = calculate_fundamental_value(fundamental, selected_methods, context, ticker_name)
                selected_value = value['output'] if type(value) == dict else value
                fundamentals_values[fundamental] = selected_value if selected_value not in "I don't know." else None
        return fundamentals_values