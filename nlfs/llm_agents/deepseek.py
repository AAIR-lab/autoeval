from nlfs.llm_agents.gpt import GPT
from nlfs import llm_agents

class DeepSeek(GPT):

    def __init__(self, model_name, temperature=0.1,
                 base_url="https://api.deepseek.com",
                 api_key_var="DEEPSEEK_API_KEY"):
        
        super(DeepSeek, self).__init__(model_name, temperature,
                                       base_url,
                                       api_key_var)
        
class DeepSeek_R1(DeepSeek):

    def __init__(self, temperature=0.1):
        super(DeepSeek_R1, self).__init__("deepseek-reasoner", temperature)
        self.input_token_cost = 0.00055
        self.output_token_cost = 0.00219

if __name__ == "__main__":

    agent = llm_agents.get_llm_agent("deepseek-r1")
    response, cost = agent.send_message("Tell me a joke")
    print(response)
    response, cost = agent.send_message("Tell me 5 more")
    print(response)

    agent.reset()
    response, cost = agent.send_message("Tell me 5 more")
    print(response)
