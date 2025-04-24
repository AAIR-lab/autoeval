from openai import OpenAI

class VLLMAgent:
    
    def __init__(self, model):
        
        self.conversation_history = []
        self.client = OpenAI(api_key="EMPTY",base_url="http://localhost:8000/v1")
        self.model = model

        self.total_input_cost = 0
        self.total_output_cost = 0
        self.input_token_cost = float("nan")
        self.output_token_cost = float("nan")

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
        
    def send_message(self,message,role='user'):
        self.conversation_history.append({"role": role, "content": message})
        response = self.client.chat.completions.create(model=self.model,messages=self.conversation_history)
        #print(response)
        self.conversation_history.append({"role":"system", "content":response.choices[0].message.content})   
        # print(generated_ids)
        # print(output)
        # print("Output = ",output)
        # print("========================\n")    
        return response.choices[0].message.content, 0

    def reset(self):
        self.conversation_history = []
    
if __name__ == "__main__":

    agent = VLLMAgent("facebook/opt-125m")
    
    response, cost = agent.send_message("Tell me a joke")
    print(response, cost)
    print("--------")
    response = agent.send_message("Tell me 5 more")
    print(response)
    print("========")

    agent.reset()
    response, cost = agent.send_message("Tell me 5 more")
    print(response, cost)