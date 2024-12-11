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
from var_cov import calc_cov_matrix, portfolio_volatility, optimize_for_target_risk

config_list=[{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]

def find_cluster(stocks: Annotated[list, "List of stocks user is interested in investing"], size: Annotated[int, "Number of similar stocks"]) -> Annotated[list, "List of stocks we suggest user to invest in"]:
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

    portfolio_risk_calculator_schema = get_function_schema(
        calc_cov_matrix,
        name="portfolio_volatility",
        description="Computes risk for the last 30 days for a portfolio cluster",
    )

    optimize_for_target_risk_schema = get_function_schema(
        optimize_for_target_risk,
        name="optimize_for_target_risk",
        description="Computes optimal weights given target",
    )

    assistant_config_entrypoint_agent={
    "assistant_id": "asst_HvA1lWZdgAvx3kk9KJknWngp",
    }

    entrypoint_agent = GPTAssistantAgent(
        name="entrypoint_agent",
        assistant_config=assistant_config_entrypoint_agent,
        instructions="""
        As 'entrypoint_agent', your primary role is to find out stocks that user/investor is interested in and then find similar stocks:

        1. Use the 'find_cluster_schema' function
        2. An input is a python list of stock tickers
        3. The second input is size and it is int which is a number of stocks in portfolio.
        4. If the size is equal to 1 it means that the user is only interested in the particular stocks he mentioned. In this case simply send back only this particular stocks as python list.
        5. Put the tickers in alphabetical order in a python list and put the weights in the same order in another python list
        6. Communicate file_path, list of tickers, date and weights
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

##    risk_calc_agent = GPTAssistantAgent(
##        name="risk_calc_agent",
##        instructions="""
##        As 'risk_calc_agent', You compute risk for the last 30 days for a cluster as follows:
##
##        1. Use the 'garman_klass_volatility' to provide last 30 days risk for investment
##        2. As the first input to the function take 'D:\Witold\Documents\Computing\LLMAgentsOfficial\Hackathon\sp500_stock_data.csv'
##        3. Second argument is python list of tickers from entrypoint_agent
##        4. Third argument is the staring date (str): The base date in 'YYYY-MM-DD' format.
##        5. You run the function for the next 15 days adding one day to the staring date each time
##        6. After each run of the function communicate the risk
##        7. If there is no results for a given date you omit it and not communicate it
##        8. Communicate to portfolio_calc_agent file_path, list of tickers, date and weights
##        """,
##        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
##        overwrite_tools=True,  # overwrite any existing tools with the ones provided
##        llm_config={
##            "config_list": config_list,
##            "tools": [risk_calculator_schema],
##        },
##    )
##
##    risk_calc_agent.register_function(
##        function_map={
##            "garman_klass_volatility": garman_klass_volatility,
##        },
##    )

    assistant_config_portfolio_calc_agent={
    "assistant_id": "asst_rFWlt3Xv2Yuw3tps6AbdT8kO",
    }

    portfolio_calc_agent = GPTAssistantAgent(
        name="portfolio_calc_agent",
        assistant_config=assistant_config_portfolio_calc_agent,
        #assistant_config=assistant_config,
        instructions="""
        As 'portfolio_calc_agent', You compute risk for each day for 15 days for a portfolio as follows:

        1. You run function "portfolio_volatility"
        2. As the first input to the function take file_path
        3. Second argument is python list of tickers from entrypoint_agent
        4. Third argument is the staring date (str): The base date in 'YYYY-MM-DD' format.
        5. Fourth argument are portfolio weights. Remember that portfolio weights must correspond to the order of tickers
        6. You run the function for the next day
        7. After each run of the function you communicate the portfolio_risk value and say "RISK VALUE" and wait for his reply before proceeding further
        8. If reply is OK you move date but one day and run 'portfolio_volatility' function again
        9. If reply is "NEW WEIGHTS" take the New Weights and you move date by one day and run 'portfolio_volatility' function again
        10. If there is no results for a given date you omit it and not communicate it
        11. Once you reach 15 days you say TERMINATE
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [portfolio_risk_calculator_schema],
        },
    )

    portfolio_calc_agent.register_function(
        function_map={
            "portfolio_volatility": portfolio_volatility,
        },
    )

    assistant_config_risk_agent={
    "assistant_id": "asst_WBAp2Sblc8nJieP4cveQ1jFf",
    }

    risk_agent = GPTAssistantAgent(
        name="risk_agent",
        assistant_config=assistant_config_risk_agent,
        instructions="""
        As 'risk_agent', You adjust portfolio weights as follows:

        1. You take portfolio risk for a given day as an input
        2. If portfolio risk is less than target risk you say OK
        3. If portfolio risk is greater than target risk you call optimize_for_target_risk function
        4. The function will produce new weights.
        5. You say "NEW WEIGHTS"and communicate updated weigths
        6. In the next step you wait to receive portfolio risk for the next day
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [optimize_for_target_risk_schema],
        },
    )

    risk_agent.register_function(
        function_map={
            "optimize_for_target_risk": optimize_for_target_risk,
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

    def state_transition(last_speaker, groupchat):
        messages = groupchat.messages

        if last_speaker is user:
            # init -> retrieve
            return entrypoint_agent
        elif last_speaker is entrypoint_agent:
            # init -> retrieve
            return portfolio_calc_agent
        elif last_speaker is portfolio_calc_agent:
            if "RISK VALUE" in messages[-1]["content"]:
                # retrieve --(execution failed)--> retrieve
                return risk_agent
            else:
                # retrieve --(execution success)--> research
                return portfolio_calc_agent
        elif last_speaker is risk_agent:
            if messages[-1]["content"] == "TERMINATE":
                # retrieve --(execution failed)--> retrieve
                return None
            else:
                # retrieve --(execution success)--> research
                return portfolio_calc_agent

    groupchat = autogen.GroupChat(agents=[user,entrypoint_agent, portfolio_calc_agent,risk_agent], messages=[], max_round=10,send_introductions=True,speaker_selection_method=state_transition)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

    result = user.initiate_chat(group_chat_manager, message=user_query,summary_method="last_msg")


if __name__ == "__main__":
    #assert len(sys.argv) > 1, 
    #main(sys.argv[1])
    main("file_path is 'D:/Witold/Documents/Computing/LLMAgentsOfficial/Hackathon/sp500_stock_data.csv', tickers are UAL and SCHO with weights 100 and 0, size is 1 and the starting date is 2024-08-01 and the target risk is 0.02")
