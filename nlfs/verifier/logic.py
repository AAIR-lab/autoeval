
from nltk.sem.logic import read_logic
from nltk.inference.prover9 import Prover9
import time
import pathlib
import re

_PROVER_9_ROOT = (pathlib.Path(__file__).parent / "../../prover9/bin").as_posix()

def expr_to_str(expression):

    string = str(expression)
    string = string.replace("|", "v")
    string = string.replace("&", "∧")
    string = string.replace("-", "¬")
    string = string.replace("all", "∀")
    string = string.replace("exists", "∃")
    string = string.lower()

    return string


def str_to_expr(string):

    if isinstance(string, str):
        string = string.lower()
        string = string.replace(" ∨ ", " or ")
        string = string.replace(" v ", " or ")
        string = string.replace(")v ", ") or ")
        string = string.replace(" v(", " or (")
        string = string.replace(")v(", ") or (")
        string = string.replace("∧", " and ")
        string = string.replace("¬", " not ")
        string = string.replace("∀", " all ")
        string = string.replace("∃", " exists ")
        string = string.replace("\n", "")
        string = string.replace("[", "(")
        string = string.replace("]", ")")
        string = string.replace("→", "->")
        string = string.replace("↔", "<->")
                

        return read_logic(string)[0]
    else:

        return string

def verify(args):

    s1, s2 = args
    timeout = float("inf")
    engine = "prover9"
    assert engine == "prover9"

    if timeout == float("inf"):

        # No timeout
        timeout = 0
    elif timeout == 0:

        # Timeout of 0 seconds
        timeout = 0.01

    try:
        s1 = str_to_expr(s1)
        s2 = str_to_expr(s2)

        start_time = time.time()
        prover = Prover9(timeout=timeout)
        prover.config_prover9(_PROVER_9_ROOT)
        result = s1.equiv(s2, prover=prover)
    except Exception as e:

        return None, None, str(e)

    return result, start_time, None

if __name__ == "__main__":

    s1 = str_to_expr("(p1 & (p3 | p1 | (-p3 & -(-p3 | p3) & -p3)))")
    s2 = "(p1 & (p3 | p1 | (-p3 & -(-p3 | p3) & -p3)))"

    print(verify(s1, s2, 0.01))