from nlfs.llm_agents.gpt import GPT_3PT5_TURBO
from nlfs.llm_agents.gpt import GPT_4
from nlfs.llm_agents.gpt import GPT_4_TURBO
from nlfs.llm_agents.gpt import GPT_4O
from nlfs.llm_agents.gpt import GPT_4O_MINI
from nlfs.llm_agents.gpt import GPT_4O1
from nlfs.llm_agents.gpt import GPT_4O1_MINI
from nlfs.llm_agents.deepseek import DeepSeek_R1
from nlfs.llm_agents.claude import Claude
from nlfs.llm_agents.vllm_agent import VLLMAgent
from nlfs.llm_agents.batch_gpt import BatchGPT_4_TURBO, BatchGPT_4O_MINI, BatchGPT_4O, BatchGPT_3PT5_TURBO,BatchGPT_4

@staticmethod
def get_llm_agent(model_name, temperature=0.1):

    if model_name == "gpt-3.5-turbo":
        return GPT_3PT5_TURBO(temperature=temperature)
    elif model_name == "gpt-4":
        return GPT_4(temperature=temperature)
    elif model_name == "gpt-4-turbo":
        return GPT_4_TURBO(temperature=temperature)
    elif model_name == "gpt-4o":
        return GPT_4O(temperature=temperature)
    elif model_name == "gpt-4o-mini":
        return GPT_4O_MINI(temperature=temperature)
    elif model_name == "gpt-4o1":
        return GPT_4O1(temperature=temperature)
    elif model_name == "gpt-4o1-mini":
        return GPT_4O_MINI(temperature=temperature)
    elif model_name == "claude":
        return Claude(temperature=temperature)    
    elif model_name == "mistral":
        return VLLMAgent("mistralai/Mistral-7B-Instruct-v0.2")
    elif model_name == "llama-3-8b":
        return VLLMAgent("meta-llama/Meta-Llama-3-8B-Instruct")
    elif model_name == "phi-3":
        return VLLMAgent("microsoft/Phi-3-medium-4k-instruct")
    elif model_name == "llama-3.1-8b":
        return VLLMAgent("meta-llama/Llama-3.1-8B-Instruct")
    elif model_name == "qwen-2.5-14b":
        return VLLMAgent("Qwen/Qwen2.5-14B-Instruct")
    elif model_name == "ministral":
        return VLLMAgent("mistralai/Ministral-8B-Instruct-2410")
    elif model_name == "granite":
        return VLLMAgent("ibm-granite/granite-3.0-8b-instruct")
    elif model_name == "deepseek-v2-lite":
        return VLLMAgent("deepseek-ai/DeepSeek-V2-Lite")
    elif model_name == "gemma-2-9b":
        return VLLMAgent("google/gemma-2-9b-it")
    elif model_name == "llama-3-70b":
        return VLLMAgent("meta-llama/Meta-Llama-3-70B-Instruct")
    elif model_name == "llama-3-8B-informalization":
        return VLLMAgent("Iddah/llama3-8B-instruct-informalization")
    elif model_name == "vicuna-13b":
        return VLLMAgent("lmsys/vicuna-13b-v1.5")
    elif model_name == "falcon-40b":
        return VLLMAgent("tiiuae/falcon-40b-instruct")
    elif model_name == "yi-1.5-34b":
        return VLLMAgent("01-ai/Yi-1.5-34B-Chat")
    elif model_name == "phi-3.5-mini":
        return VLLMAgent("microsoft/Phi-3.5-mini-instruct")
    elif model_name == "llama-3.2-1b":
        return VLLMAgent("meta-llama/Llama-3.2-1B-Instruct")
    elif model_name == "llama-3.1-8b-prop-logic-lora":
        return VLLMAgent("llama-3.1-8b-prop-logic-lora")
    elif model_name == "qwen-2.5-1.5b":
        return VLLMAgent("Qwen/Qwen2.5-1.5B-Instruct")
    elif model_name == "deepseek-r1":
        return DeepSeek_R1(temperature=temperature)
    else:
        raise Exception("Invalid model name")

def get_batch_llm_agent(model_name, temperature=0.1):
    if model_name == "gpt-3.5-turbo":
        return BatchGPT_3PT5_TURBO(temperature=temperature)
    elif model_name == "gpt-4":
        return BatchGPT_4(temperature=temperature)
    elif model_name == "gpt-4-turbo":
        return BatchGPT_4_TURBO(temperature=temperature)
    elif model_name == "gpt-4o":
        return BatchGPT_4O(temperature=temperature)
    elif model_name == "gpt-4o-mini":
        return BatchGPT_4O_MINI(temperature=temperature)
    else:
        raise Exception("Invalid model name")
