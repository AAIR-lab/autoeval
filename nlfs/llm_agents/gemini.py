
GOOGLE_API_KEY = 'Invalid'

import google.generativeai as genai
import google
import backoff
genai.configure(api_key=GOOGLE_API_KEY)



def _print_backoff_info(details):
    pass
    # print("Total backoff tries: {tries}".format(**details))

class Gemini:
    def __init__(self,model_name='gemini-1.5-pro-001'):

        self.model = model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.total_input_cost = 0
        self.total_output_cost = 0
        self.input_token_cost = 0.0035
        self.output_token_cost = 0.007



    @backoff.on_exception(backoff.expo,
                          google.api_core.exceptions.ResourceExhausted,
                          max_tries=20,
                          jitter=None,
                          on_backoff=_print_backoff_info)

    def _send(self):

            input_tokens = self.model.count_tokens(self.conversation_history).total_tokens
            response = self.model.generate_content(self.conversation_history)
            output_tokens = self.model.count_tokens(response.candidates[0].content).total_tokens
         
            return response, input_tokens, output_tokens

    def send_message(self, message):

        self.conversation_history.append({'role':'user', 'parts': [message]})

    
        response, input_tokens, output_tokens = self._send()

        self.conversation_history.append(response.candidates[0].content)

        cost = self.update_costs(input_tokens,output_tokens)


        return response.candidates[0].content.parts[0].text, cost

    def reset(self):
        self.conversation_history = []

    def get_conversation_history(self):
        return self.conversation_history
        
    def update_costs(self, input_tokens, output_tokens):

        input_cost = (input_tokens * self.input_token_cost) / 1000.0
        output_cost = (output_tokens * self.output_token_cost) / 1000.0

        self.total_input_cost += input_cost
        self.total_output_cost += output_cost

        return input_cost + output_cost

        return self.total_input_cost, self.total_output_cost

    def get_total_cost(self):

        return self.total_input_cost + self.total_output_cost

    def get_conversation_history(self):
        return self.conversation_history


# Example of creating and using the class
if __name__ == '__main__':
    agent = Gemini()
    
    for i in range(1):
        response, cost = agent.send_message("Tell me a joke")
        print(response,cost)
        response, cost = agent.send_message("Tell me 5 more")
        print(response,cost)

    # agent.reset()
    # response, cost = agent.send_message("Tell me 5 more")
    # print(response)
