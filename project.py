import os
import openai

from pathlib import Path

from memgpt.agent import Agent as _Agent

from typing import Callable, Optional, List, Dict, Union, Any, Tuple

from memgpt.persistence_manager import LocalStateManager
import memgpt.constants as constants
import memgpt.utils as utils
import memgpt.presets.presets as presets
from memgpt.config import AgentConfig
from memgpt.interface import CLIInterface as interface


from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

# Point to your config file (or remove this line if you just want to use the default)
hallo = os.environ['MEMGPT_CONFIG_PATH'] = Path.home().joinpath('.memgpt').joinpath('config').as_posix()


# Necessary to avoid errors when LLM is invoked
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = 'https://api.openai.com/v1'

# persona_desc = utils.get_persona_text(constants.DEFAULT_PERSONA)
persona_desc = utils.get_persona_text("1440.txt")
user_desc = utils.get_human_text("Boesen.txt")



# Create an AgentConfig option from the inputs
agent_config = AgentConfig(
    name="agent_4",
    persona=persona_desc,
    human=user_desc,
    preset="memgpt_chat",
    model="gpt-4",
    model_endpoint_type="openai",
    model_endpoint="https://api.openai.com/v1",
    context_window=8192,
)

skip_verify = agent_config.model != "gpt-4"

NEW_AGENT = False


if NEW_AGENT:
    persistence_manager = LocalStateManager(agent_config)

    memgpt_agent = presets.use_preset(
        preset_name=agent_config.preset,
        agent_config=agent_config,
        model=agent_config.model,
        persona=agent_config.persona,
        human=agent_config.human,
        interface=interface,
        persistence_manager=persistence_manager,
    )

    memgpt_agent.step(user_message="Hi my name is Bill Gates.  I love the Windows OS!", first_message=True, skip_verify=skip_verify)
    memgpt_agent.step(user_message="I am 68 years old, born October 28, 1955", first_message=False, skip_verify=skip_verify)
    memgpt_agent.save()
else:
    memgpt_agent = _Agent.load_agent(interface, agent_config)
    memgpt_agent.step(user_message="What have i told you that i like to drink?", first_message=True, skip_verify=skip_verify)
    # memgpt_agent.step(user_message="What is my age?", first_message=False, skip_verify=skip_verify)