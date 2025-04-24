import anthropic
import backoff
import requests
from nlfs.llm_agents import llm_agent

MODEL ="claude-3-sonnet-20240229"

def _print_backoff_info(details):
    pass
    # print("Total backoff tries: {tries}".format(**details))


class ClaudeRateLimitError(Exception):
    pass

class Claude:

    def __init__(self, temperature=0.1):

        api_key = llm_agent.get_api_key("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        
        #costs from anthropic.com/api --> per 1k tokens
        self.total_input_cost = 0
        self.total_output_cost = 0
        self.input_token_cost = 0.015
        self.output_token_cost = 0.075     
        self.temperature = temperature

    def update_costs(self, input_tokens, output_tokens):

        input_cost = (input_tokens * self.input_token_cost) / 1000.0
        output_cost = (output_tokens * self.output_token_cost) / 1000.0

        self.total_input_cost += input_cost
        self.total_output_cost += output_cost

        return input_cost + output_cost

    def get_costs(self):

        return self.total_input_cost, self.total_output_cost

    def get_total_cost(self):

        return self.total_input_cost + self.total_output_cost

        
    def reset(self):
        self.conversation_history = []    
        

    @backoff.on_exception(backoff.expo,
                         ClaudeRateLimitError,
                          max_tries=20,
                          jitter=None,
                          on_backoff=_print_backoff_info)    


    def _send(self,model,max_tokens):
        try:
            response = self.client.messages.create(
                model = model,
                max_tokens = max_tokens,
                messages=self.conversation_history,
                temperature = self.temperature   
            )

            return response
        except Exception as e:
            if e.status_code==429:
                raise ClaudeRateLimitError
            else:
                raise e


    def send_message(self,message_text,model=MODEL,max_tokens=1024,role='user'):
        
        self.conversation_history.append({"role": role, "content": message_text})

        response = ""


        response = self._send(model,max_tokens)


        self.conversation_history.append({"role":"assistant", "content":response.content[0].text})        
        cost = self.update_costs(response.usage.input_tokens,
                          response.usage.output_tokens)
        
        return response.content[0].text, cost
    
    def get_conversation_history(self):

        return self.conversation_history    
    

if __name__ == "__main__":
    # from fol_generator import FOLGenerator

    # variables = ["x", "y", "z"]
    # constants = ["William", "Alice"]
    # predicates = {"MutualFriends":3,"ShareHobby":2,"Tutors":2}
    # generator = FOLGenerator(3,variables,constants,predicates)
    # num_ops = 5

    # # try:
    # test_model = Claudeß()

    # for x in range(1):
    #     formula = generator.generate_formula(num_ops).string_val
    #     print(x)
    #     print(test_model.send_message("Please explain the following first order logic formula such that it's both understandable to a person but can be used to recreate the formula. " + formula))
    #     print("------------------------------------------")

    agent = Claude()
    response, cost = agent.send_message("Tell me a joke")
    print(response, cost)
    response = agent.send_message("Tell me 5 more")
    print(response)

    agent.reset()
    response, cost = agent.send_message("Tell me 5 more")
    print(response, cost)


    # except Exception as e:
    #     print(e)
