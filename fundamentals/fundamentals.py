import logging
from typing import Optional, Any

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool,AgentType
from langchain.embeddings import OpenAIEmbeddings
from  create_pinecone_db import vectorstore
from langchain.chains.llm import LLMChain



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
    You are an agent that answers questions by running Python code using the tool `python_repl`.

    Your job is to write a Python command and run it using this tool. You must ALWAYS use `print(...)` to display the final result.

    Task:
    Use Python to print the result of `dir(yf.Ticker)`.

    Example:
    Action: python_repl
    Action Input: 
    import yfinance as yf
    print(dir(yf.Ticker))
    """

    format_with_instructions = """
    format data: {data} as mentioned below

    {format_instructions}

    """

    get_important_methods = """
    Here are the important attributes and methods {all_methods} of yfinance.Ticker among them Identify the methods which we can use to calculate {fundamental}.
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

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini" )

    agent = create_python_agent(
        llm,
        tool=repl_tool,
        verbose=True,
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
    def get_context_from_methods(cls, methods: tuple) -> dict:
        selected_methods_context = []
        for method in methods[0]:
            print(method)
            value = vectorstore.max_marginal_relevance_search(method, k=1)
            print("VALUE:----", value)
            selected_methods_context.append(value[0].page_content)
        combined_context = "\n------\n".join(selected_methods_context)
        return {'combined_context': combined_context}


    @classmethod
    def get_yfianance_function_list(cls):
        logging.info("getting methods supported by yfinance..........")
        methods = ResponseSchema(name="methods",
                                 description="methods supported by yfinance python librabry which will be used to identify the fundamental later",
                                 type="list")
        response_schema = [methods]
        output_parser_methods = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser_methods.get_format_instructions()
        promt_format_instructions_template = ChatPromptTemplate.from_template(cls.format_with_instructions)
        print_values_prompt = ChatPromptTemplate.from_template(cls.get_function_prompt)
        logging.info("trying to get_yfianance_function_list")

        map_run = RunnableMap(
            {
                'data': lambda x: cls.invoke_agent(x['print_values_prompt'], 2),
                'format_instructions': lambda x: x['format_instructions']
            }
        )
        chain = map_run | promt_format_instructions_template | cls.llm | output_parser_methods
        value = chain.invoke({'print_values_prompt': print_values_prompt, 'format_instructions': format_instructions})
        return  value['methods']


    @classmethod
    def get_company_yfinance_ticker(cls, text: str) -> str:

        ticker = ResponseSchema(name="output_value",
                                description="The ticker name supported by yfianance python librabry")
        response_schema = [ticker]

        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()

        prompt_input_value_template = ChatPromptTemplate.from_template(cls.find_ticker_value)
        ticker_chain = LLMChain(llm=cls.llm, prompt=prompt_input_value_template, output_parser=output_parser)
        response = ticker_chain.invoke({'input_value': text, 'format_instructions': format_instructions})
        return response['text']['output_value']

    @classmethod
    def get_top_n_fundamentals(cls, count: int) -> list:
        """
        :param count: the number of desired fundamentals
        :return: list of fundamentals
        """
        fundametal_list = ResponseSchema(name="fundamentals",
                                         description="list of fundamentals separated by comma in list format ",
                                         type="list")
        response_schema = [fundametal_list]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        prompt_find_fundamentals_template = ChatPromptTemplate.from_template(cls.promt_find_fundamentals)
        fundamental_chain = LLMChain(prompt=prompt_find_fundamentals_template, llm=cls.llm, output_parser=output_parser)
        fundamentals = fundamental_chain.invoke({'n': count, 'format_instructions': format_instructions})
        return fundamentals['text']['fundamentals']


    @classmethod
    def get_fundamenta_values(cls, fundamentals: list, ticker_name: str) -> dict:

        """

        :param fundamentals:  list of fundamentals to be calculated
        :param ticker_name:  ticker name as per yfianance
        :return:  dictionary of fundamentals
        """

        fundamentals_values = dict()

        y_finance_methods = cls.get_yfianance_function_list()

        def get_method_details(x):

            methods = chain_imp_methods.invoke({'fundamental': x['fundamental'], 'all_methods': x['all_methods'],
                                                'format_instructions': x['format_instructions']})['text']['methods'],
            logging.info("Selected methods:", methods)
            return {'methods': methods,
                    'ticker_name': x['ticker_name'],
                    'fundamental': x['fundamental'],
                    'combined_context': cls.get_context_from_methods(methods)}

        def invoke_agent_two_retries(x):
            return cls.invoke_agent(x, 2),

        logging.info("fundamentals:",fundamentals)
        for fundamental in fundamentals:
            print("Getting value of fundamental: ", fundamental)

            methods = ResponseSchema(name="methods",
                                     description="methods or methods supported by yfianance python librabry which can be used to identify or calculate the fundamental value, include max 3 methods",
                                     type="list")

            response_schema = [methods]
            output_parser = StructuredOutputParser.from_response_schemas(response_schema)
            format_instructions_get_methods = output_parser.get_format_instructions()
            get_important_methods_prompt = ChatPromptTemplate.from_template(cls.get_important_methods)
            chain_imp_methods = LLMChain(prompt=get_important_methods_prompt, llm=cls.llm, output_parser=output_parser)
            calculate_fundamental_prompt = ChatPromptTemplate.from_template(cls.calculate_fundamental_template)
            final_chain = RunnableLambda(get_method_details) | calculate_fundamental_prompt | invoke_agent_two_retries
            value = final_chain.invoke({'fundamental': fundamental,
                                        'all_methods': y_finance_methods,
                                        'format_instructions': format_instructions_get_methods,
                                        'ticker_name': ticker_name})
            logging.info("value: ", value)

            selected_value = value[0]['output'] if type(value[0]) == dict else value

            fundamentals_values[fundamental] = selected_value if selected_value not in "I don't know." else None

            # Retry if the llm agent returns I don't know
            if fundamentals_values[fundamental] is None:
                value = final_chain.invoke({'fundamental': fundamental,
                                            'all_methods': y_finance_methods,
                                            'format_instructions': format_instructions_get_methods,
                                            'ticker_name': ticker_name})
                selected_value = value[0]['output'] if type(value[0]) == dict else value
                fundamentals_values[fundamental] = selected_value if selected_value not in "I don't know." else None

        return fundamentals_values