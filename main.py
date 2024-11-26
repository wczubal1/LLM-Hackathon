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

def monitor_optimal_risk(cluster: Annotated[list, "Computes risk for the last 30 days for a cluster"]):
    """
    Will monitor risk daily on the position

    Args:
        cluster (list): "Computes risk for the last 30 days for a cluster"

    Returns:
        rosk for the investments
    """


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
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
            "tools": [fetch_restaurant_data_schema],
        },
    )

    entrypoint_agent.register_function(
        function_map={
            "find_cluster": find_cluster,
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
            "tools": [calculate_score_schema],
        },
    )

    risk_agent.register_function(
        function_map={
            "monitor_optimal_risk": monitor_optimal_risk,
        },
    )

    trade_agent = GPTAssistantAgent(
        name="trade_agent",
        instructions="""
        Adjusts positions based on risk
        """,
        overwrite_instructions=True,  # overwrite any existing instructions with the ones provided
        overwrite_tools=True,  # overwrite any existing tools with the ones provided
        llm_config={
            "config_list": config_list,
        },
    )

    trade_agent.register_function(
        function_map={
            "": ,
        },
    )

    groupchat = autogen.GroupChat(agents=[user, entrypoint_agent, risk_agent, trade_agent], messages=[], max_round=15)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

    result = user.initiate_chat(group_chat_manager, message=user_query,summary_method="last_msg")


if __name__ == "__main__":
    assert len(sys.argv) > 1, 
    main(sys.argv[1])
