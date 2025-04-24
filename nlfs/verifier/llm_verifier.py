from nlfs import llm_agents
import time

def generate_prompt(fs, llm_fs, data_type):
    if data_type in ["ksat", "plogic"]:
        problem_type = "propositional logic"
    elif data_type in ["fol", "fol_human"]:
        problem_type = "first-order logic"
    elif data_type in ["regex"]:
        problem_type = "regex"
    else:
        raise Exception("Unknown data type in Equiv Prompt: %s" % data_type)
    prompt = [
        "Your task is to say whether two %s formulae are equivalent. The first formula will appear right after [FORMULA 1] and the second after [FORMULA 2]." % problem_type,
        "Give an explanation followed by a yes or no answer. The answer must show up at the end with the format \"[Answer]\" followed by either a yes or no.",
        "[Formula 1] %s" % fs,
        "[Formula 2] %s" % llm_fs
    ]

    return "\n".join(prompt)

def check_answer(nl):
    nl = nl.lower()
    answer_location = nl.rfind("answer")

    if answer_location == -1:
        return False, None
    nl = nl[answer_location:]
    yes_found = "yes" in nl
    no_found = "no" in nl
    if yes_found and not no_found:
        return True, True
    elif no_found and not yes_found:
        return True, False
    else:
        return False, None

def verify(args):
    fs, llm_fs, llm_agent_name, data_type = args
    start_time = time.time()
    try:
        llm_agent = llm_agents.get_llm_agent(llm_agent_name)
        fs_to_nl = generate_prompt(fs,llm_fs,data_type)
        nl, _ = llm_agent.send_message(fs_to_nl)

    except Exception as e:
        return None, fs_to_nl, None, None, str(e)
    found_answer, result = check_answer(nl)
    if not found_answer:
        return None, fs_to_nl, nl, None, "LLM did not return a recognizable yes or no answer."
    return result, fs_to_nl, nl, start_time, None

if __name__ == "__main__":

    s1 = "(p1 & (p3 | p1 | (-p3 & -(-p3 | p3) & -p3)))"
    s2 = "(-p1 & (p3 | p1 | (p3 & -(-p3 | p3) & -p3)))"

    print(verify((s1, s2, "plogic","gpt-3.5-turbo")))