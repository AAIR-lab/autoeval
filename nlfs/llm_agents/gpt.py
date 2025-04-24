# import pytholog as pl
import openai
import backoff
from nlfs import llm_agents
from nlfs.llm_agents import llm_agent

def _print_backoff_info(details):
    print("Total backoff tries: {tries}".format(**details))

class GPT:

    def __init__(self, model_name, temperature=0.1,
                 base_url="https://api.openai.com/v1",
                 api_key_var="OPENAI_API_KEY"):

        api_key = llm_agent.get_api_key(api_key_var)

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.conversation_history = []
        self.temperature = temperature
        self.total_input_cost = 0
        self.total_output_cost = 0
        self.input_token_cost = float("nan")
        self.output_token_cost = float("nan")

    def reset(self):

        self.conversation_history = []
    @backoff.on_exception(backoff.expo,
                          openai.RateLimitError,
                          max_tries=20,
                          jitter=None,
                          on_backoff=_print_backoff_info)
    def _send(self, messages):

        return self.client.chat.completions.create(model=self.model_name,
                                                   messages=messages,
                                                   temperature=self.temperature)

    def _send_messages(self, messages):

        response = self._send(messages)
        if response is None:
            raise Exception("Max attempts for api access reached")
        elif response.choices[0].finish_reason != "stop":
            raise Exception("Invalid finish reason [%s]" % (
                response.choices[0].finish_reason))

        return response

    def update_costs(self, input_tokens, output_tokens):

        input_cost = (input_tokens * self.input_token_cost) / 1000.0
        output_cost = (output_tokens * self.output_token_cost) / 1000.0

        self.total_input_cost += input_cost
        self.total_output_cost += output_cost

        return input_cost + output_cost

    def get_conversation_history(self):

        return self.conversation_history

    def get_costs(self):

        return self.total_input_cost, self.total_output_cost

    def get_total_cost(self):

        return self.total_input_cost + self.total_output_cost

    def send_message(self, message_text, role='user'):

        new_message = {"role": role, "content": message_text}
        self.conversation_history.append(new_message)

        response = self._send_messages(self.conversation_history)
        response_role = response.choices[0].message.role
        response_text = response.choices[0].message.content

        self.conversation_history.append({'role': response_role,
                                          'content': response_text})


        cost = self.update_costs(response.usage.prompt_tokens,
                          response.usage.completion_tokens)

        return response.choices[0].message.content, cost

class GPT_3PT5_TURBO(GPT):

    def __init__(self, temperature=0.1):

        super(GPT_3PT5_TURBO, self).__init__("gpt-3.5-turbo-0125",
                                             temperature)
        self.input_token_cost = 0.0005
        self.output_token_cost = 0.0015

class GPT_4(GPT):

    def __init__(self, temperature=0.1):
        super(GPT_4, self).__init__("gpt-4-0613",
                                             temperature)
        self.input_token_cost = 0.03
        self.output_token_cost = 0.06

class GPT_4_TURBO(GPT):

    def __init__(self, temperature=0.1):
        super(GPT_4_TURBO, self).__init__("gpt-4-turbo-2024-04-09",
                                             temperature)
        self.input_token_cost = 0.01
        self.output_token_cost = 0.03

class GPT_4O(GPT):

    def __init__(self, temperature=0.1):
        super(GPT_4O, self).__init__("gpt-4o-2024-08-06",
                                             temperature)
        self.input_token_cost = 0.005
        self.output_token_cost = 0.015

class GPT_4O_MINI(GPT):
    def __init__(self, temperature=0.1):
        super(GPT_4O_MINI, self).__init__("gpt-4o-mini",
                                             temperature)
        self.input_token_cost = 0.00015
        self.output_token_cost = 0.000600    

class GPT_4O1(GPT):
    def __init__(self, temperature=0.1):
        super(GPT_4O1, self).__init__("o1-preview-2024-09-12",
                                             temperature=1)
        self.input_token_cost = 0.015
        self.output_token_cost = 0.060

class GPT_4O1_MINI(GPT):
    def __init__(self, temperature=0.1):
        super(GPT_4O1_MINI, self).__init__("o1-mini-2024-09-12",
                                             temperature=1)
        self.input_token_cost = 0.003
        self.output_token_cost = 0.012

if __name__ == "__main__":

    agent = llm_agents.get_llm_agent("gpt-3.5-turbo-0125")
    response, cost = agent.send_message("Tell me a joke")
    print(response)
    response, cost = agent.send_message("Tell me 5 more")
    print(response)

    agent.reset()
    response, cost = agent.send_message("Tell me 5 more")
    print(response)
