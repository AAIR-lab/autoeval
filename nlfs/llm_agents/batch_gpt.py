import time
import openai
import os
import datetime
import json
# from nlfs import llm_agents
# from nlfs.llm_agents import llm_agent
from pathlib import Path

class BatchGPT:

    def __init__(self, model_name, temperature=0.1, log_path="results/batch_log"):

        api_key = llm_agent.get_api_key("OPENAI_API_KEY")

        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.total_input_cost = 0
        self.total_output_cost = 0
        self.input_token_cost = float("nan")
        self.output_token_cost = float("nan")

        self.log_path = os.path.join(os.path.abspath(os.path.expanduser(log_path)),model_name)
        Path(self.log_path).mkdir(parents=True,exist_ok=True)

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

    def start_batch(self,messages, role='user'):
        assert(len(messages) < 50000)
        meta_log = {}
        meta_log["creation_time"] = str(datetime.datetime.now())

        batch_number = len(os.listdir(self.log_path))
        batch_name = "batch_%i"%(batch_number)
        batch_path = os.path.join(self.log_path,batch_name)
        Path(batch_path).mkdir(parents=False,exist_ok=False)
        meta_file_path = os.path.join(batch_path,"meta.json")
        input_json_path = os.path.join(batch_path,"input.json")
        error_json_path = os.path.join(batch_path,"error.json")

        #build input json

        with open(input_json_path,"w+") as input_messages:
            for current_index, current_message in enumerate(messages):
                request_json = {}
                request_json["custom_id"] = "request-%i"%(current_index)
                request_json["method"] = "POST"
                request_json["url"] = "/v1/chat/completions"
                request_json["body"] = {
                    "model" : self.model_name,
                    "messages" : [{"role":role,
                                   "content": current_message}],
                    "temperature" : self.temperature
                }
                input_messages.write(json.dumps(request_json))
                input_messages.write("\n")

        #upload json
        batch_input_file = self.client.files.create(file=open(input_json_path, "rb"),
                                                    purpose="batch")
        #start batch
        batch_object = self.client.batches.create(input_file_id=batch_input_file.id,
                                                  endpoint="/v1/chat/completions",
                                                  completion_window="24h",
                                                  metadata={
                                                      "batch_number": str(batch_number)
                                                      })
        
        meta_log["status"] = batch_object.status
        meta_log["id"] = batch_object.id
        meta_log["errors"] = batch_object.errors
        meta_log["output_file_id"] = None
        if batch_object.error_file_id is not None:
            error_response = self.client.files.content(batch_object.output_file_id)
            with open(error_json_path, "wb+") as output_file:
                output_file.write(error_response.content)

        with open(meta_file_path, "w+") as meta_file:
            json.dump(meta_log,meta_file, indent=4, ensure_ascii=False)

        return batch_name

    def __update_meta(self, meta_log, error_path):
        batch_object = self.client.batches.retrieve(meta_log["id"])
        meta_log["status"] = batch_object.status
        meta_log["errors"] = batch_object.errors
        meta_log["output_file_id"] = batch_object.output_file_id
        if batch_object.error_file_id is not None:
            error_response = self.client.files.content(batch_object.output_file_id)
            with open(error_path, "wb+") as output_file:
                output_file.write(error_response.content)  

    def check_batch_status(self, batch_name):
        batch_path = os.path.join(self.log_path,batch_name)
        meta_file_path = os.path.join(batch_path,"meta.json")
        error_json_path = os.path.join(batch_path,"error.json")

        with open(meta_file_path,"r") as meta_file:
            meta_log = json.load(meta_file)

        self.__update_meta(meta_log,error_json_path)
        with open(meta_file_path, "w+") as meta_file:
            json.dump(meta_log,meta_file, indent=4, ensure_ascii=False)

        return meta_log["status"]

    def extract_batch_data(self, batch_name):
        batch_path = os.path.join(self.log_path,batch_name)
        meta_file_path = os.path.join(batch_path,"meta.json")
        error_json_path = os.path.join(batch_path,"error.json")
        output_json_path = os.path.join(batch_path,"output.json")

        with open(meta_file_path,"r") as meta_file:
            meta_log = json.load(meta_file)

        if meta_log["status"] != "completed":
            self.__update_meta(meta_log,error_json_path)
        if meta_log["status"] != "completed":
            raise Exception("Batch \"%s\" has not completed."%(batch_name))
        
        file_response = self.client.files.content(meta_log["output_file_id"])
        with open(output_json_path, "wb+") as output_file:
            output_file.write(file_response.content)

        response_mapping = {}
        current_index = 0
        with open(output_json_path,"r") as input_file:
            for current_line in input_file:
                temp_json = json.loads(current_line.strip())
                response_mapping[temp_json["custom_id"]] = current_index
                current_index += 1

        responses = [""] * current_index
        costs = [0.0] * current_index
        with open(output_json_path,"r") as output_file:
            for current_line in output_file:
                temp_json = json.loads(current_line.strip())
                responses[response_mapping[temp_json["custom_id"]]]=temp_json["response"]["body"]["choices"][0]["message"]["content"]
        
                costs[response_mapping[temp_json["custom_id"]]] = self.update_costs(temp_json["response"]["body"]["usage"]["prompt_tokens"],temp_json["response"]["body"]["usage"]["completion_tokens"])
        return responses, costs             


    # def send_messages(self, messages, role='user'):
    #     # meta_log = {}
    #     # meta_log["creation_time"] = str(datetime.datetime.now())

    #     # batch_number = len(os.listdir(self.log_path))
    #     # batch_name = "batch_%i"%(batch_number)
    #     # batch_path = os.path.join(self.log_path,batch_name)
    #     # Path(batch_path).mkdir(parents=False,exist_ok=False)
    #     # meta_file_path = os.path.join(batch_path,"meta.txt")
    #     # input_json_path = os.path.join(batch_path,"input.json")
    #     # output_json_path = os.path.join(batch_path,"output.json")

    #     # #build input json

    #     # with open(input_json_path,"w+") as input_messages:
    #     #     for current_index, current_message in enumerate(messages):
    #     #         request_json = {}
    #     #         request_json["custom_id"] = "request-%i"%(current_index)
    #     #         request_json["method"] = "POST"
    #     #         request_json["url"] = "/v1/chat/completions"
    #     #         request_json["body"] = {
    #     #             "model" : self.model_name,
    #     #             "messages" : [{"role":role,
    #     #                            "content": current_message}],
    #     #             "temperature" : self.temperature
    #     #         }
    #     #         input_messages.write(json.dumps(request_json))
    #     #         input_messages.write("\n")

    #     # #upload json
    #     # batch_input_file = self.client.files.create(file=open(input_json_path, "rb"),
    #     #                                             purpose="batch")
    #     # #start batch
    #     # batch_object = self.client.batches.create(input_file_id=batch_input_file.id,
    #     #                                           endpoint="/v1/chat/completions",
    #     #                                           completion_window="24h",
    #     #                                           metadata={
    #     #                                               "batch_number": str(batch_number)
    #     #                                               })
    #     #check batch
    #     batch_object = self.client.batches.retrieve('batch_unZsLMh2wNugsczEmBXKfMoI')
    #     while batch_object.status != "completed":
    #         time.sleep(self.check_time)
    #         batch_object = self.client.batches.retrieve(batch_object.id)

    #     #complete message
    #     file_response = self.client.files.content(batch_object.output_file_id)
    #     with open(output_json_path, "wb+") as output_file:
    #         output_file.write(file_response.content)

    #     responses = []
    #     with open(output_json_path,"r") as output_file:
    #         for current_line in output_file:
    #             temp_json = json.loads(current_line.strip())
    #             responses.append(temp_json["response"]["body"]["choices"][0]["message"]["content"])

    #     cost = 0.0
    #     return responses, cost

class BatchGPT_3PT5_TURBO(BatchGPT):

    def __init__(self, temperature=0.1):

        super(BatchGPT_3PT5_TURBO, self).__init__("gpt-3.5-turbo-0125",
                                             temperature)
        self.input_token_cost = 0.00025
        self.output_token_cost = 0.00075

class BatchGPT_4(BatchGPT):

    def __init__(self, temperature=0.1):
        super(BatchGPT_4, self).__init__("gpt-4-0613",
                                             temperature)
        self.input_token_cost = 0.015
        self.output_token_cost = 0.03

class BatchGPT_4_TURBO(BatchGPT):

    def __init__(self, temperature=0.1):
        super(BatchGPT_4_TURBO, self).__init__("gpt-4-turbo-2024-04-09",
                                             temperature)
        self.input_token_cost = 0.005
        self.output_token_cost = 0.015

class BatchGPT_4O(BatchGPT):

    def __init__(self, temperature=0.1):
        super(BatchGPT_4O, self).__init__("gpt-4o-2024-08-06",
                                             temperature)
        self.input_token_cost = 0.0025
        self.output_token_cost = 0.0075

class BatchGPT_4O_MINI(BatchGPT):
    def __init__(self, temperature=0.1):
        super(BatchGPT_4O_MINI, self).__init__("gpt-4o-mini",
                                             temperature)
        self.input_token_cost = 0.0025
        self.output_token_cost = 0.0075    

if __name__ == "__main__":
    test = BatchGPT_3PT5_TURBO()
    test_messages = [
        "If the only yellow fruit when ripe in my friend's fruit orchard are lemons and I see a yellow fruit, what can I conclude?",
        "How fast is the speed of light?",
        "Tell me 5 computer science jokes."
    ]

    batch_name = test.start_batch(test_messages)
    print("Batch name: %s"%(batch_name))
    print("Sleeping for 30 seconds")
    time.sleep(30)
    batch_status = test.check_batch_status(batch_name)
    print("Batch status: %s"%(batch_status))
    while batch_status != "completed":
        print("Sleeping for 30 seconds")
        time.sleep(30)
        batch_status = test.check_batch_status(batch_name)        
        print("Batch status: %s"%(batch_status))
    
    responses, costs = test.extract_batch_data(batch_name)

    print("Responses:")
    for current_question, current_response, current_cost in zip(test_messages,responses, costs):
        print("Question: %s"%(current_question))
        print("Response: %s"%(current_response))
        print("Cost: %f\n"%(current_cost))
