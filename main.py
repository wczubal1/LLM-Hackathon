import logging
import os
import sys
import autogen
from autogen import ConversableAgent, UserProxyAgent
from openai import OpenAI
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.function_utils import get_function_schema
from typing import Annotated, Dict, List
import numpy as np
from pathlib import Path
#import SP500data
from G_K_vol import garman_klass_volatility

config_list=[{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

def find_cluster(stocks: Annotated[list, "List of stocks user is interested in investing"], size: Annotated[int, "Number of similar stocks"]):
    """
    This will take as an input list of stocks that user is interested in. Will run encoder decoder neural network dinding stocks that recently
    behaved in a similar way
    Args:
        stocks (list): "List of stocks user is interested in investing"

    Returns:
        list: list of stocks that are suggested for a client from similar cluster
    """
    return stocks

def monitor_optimal_risk(cluster: Annotated[list, "Computes risk for the last 30 days for a cluster"]):
    """
    Will monitor risk daily on the position

    Args:
        cluster (list): "Computes risk for the last 30 days for a cluster"

    Returns:
        rosk for the investments
    """
    pass


def main(user_query: str):

    llm_config = {
        "config_list": config_list,
    }

    # the main entrypoint/supervisor agent
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "last_n_messages": 1,
            "use_docker": False,
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    find_cluster_schema = get_function_schema(
        find_cluster,
        name="find_cluster",
        description="Finds cluster of stocks for a suggested investment",
    )

    risk_calculator_schema = get_function_schema(
        garman_klass_volatility,
        name="garman_klass_volatility",
        description="Computes risk for the last 30 days for a cluster",
    )

    monitor_optimal_risk_schema = get_function_schema(
        monitor_optimal_risk,
        name="monitor_optimal_risk",
        description="Computes risk for the last 30 days for a cluster",
    )


    entrypoint_agent = GPTAssistantAgent(
        name="entrypoint_agent",
        instructions="""
        As 'entrypoint_agent', your primary role is to find out stocks that user/investor is interested in and then find similar stocks:

        1. Use the 'find_cluster_schema' function
        2. An input is a python list of stock tickers
        3. The second input is size and it is int which is a number of stocks in portfolio.
        4. If the size is equal to 1 it means that the user is only interested in the particular stock he mentioned. In this case simply send back only this particular stock as python list.
        5. Communicate to risk_calc_agent file_path, list of tickers, and a date
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [find_cluster_schema],
        },
    )

    entrypoint_agent.register_function(
        function_map={
            "find_cluster": find_cluster,
        },
    )

    risk_calc_agent = GPTAssistantAgent(
        name="risk_calc_agent",
        instructions="""
        As 'risk_calc_agent', You compute risk for the last 30 days for a cluster as follows:

        1. Use the 'garman_klass_volatility' to provide last 30 days risk for investment
        2. As the first input to the function take 'D:\Witold\Documents\Computing\LLMAgentsOfficial\Hackathon\sp500_stock_data.csv'
        3. Second argument is python list of tickers from entrypoint_agent
        4. Third argument is the staring date (str): The base date in 'YYYY-MM-DD' format.
        5. You run the function for the next 15 days adding one day to the staring date each time
        6. After each run of the function communicate the risk to risk_agent
        7. If there is no results for a given date you omit it and not communicate it
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [risk_calculator_schema],
        },
    )

    risk_calc_agent.register_function(
        function_map={
            "garman_klass_volatility": garman_klass_volatility,
        },
    )

    risk_agent = GPTAssistantAgent(
        name="risk_agent",
        instructions="""
        As 'risk_agent', You compute risk for the last 30 days for a cluster as follows:

        1. Use the 'monitor_optimal_risk' to provide last 30 days risk for investment
        2. If risk is outside of bands contact 'trade_agent' and tell him adjustements are needed
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [monitor_optimal_risk_schema],
        },
    )

    risk_agent.register_function(
        function_map={
            "monitor_optimal_risk": monitor_optimal_risk,
        },
    )

##    trade_agent = GPTAssistantAgent(
##        name="trade_agent",
##        instructions="""
##        Adjusts positions based on risk
##        """,
##        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
##        overwrite_tools=True,  # overwrite any existing tools with the ones provided
##        llm_config={
##            "config_list": config_list,
##        },
##    )
##
##    trade_agent.register_function(
##        function_map={
##            "": ,
##        },
##    )

    groupchat = autogen.GroupChat(agents=[user, entrypoint_agent, risk_calc_agent], messages=[], max_round=15)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

    result = user.initiate_chat(group_chat_manager, message=user_query,summary_method="last_msg")


if __name__ == "__main__":
    #assert len(sys.argv) > 1, 
    #main(sys.argv[1])
    main("file_path is 'D:/Witold/Documents/Computing/LLMAgentsOfficial/Hackathon/sp500_stock_data.csv', ticker is NVDA, size is 1 and the starting date is 2024-03-01")
