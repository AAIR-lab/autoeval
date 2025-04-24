import contextlib
import tempfile
import networkx as nx
from networkx import isomorphism as iso
import time
import subprocess
import pathlib

REGEX_PATH = (pathlib.Path(__file__).parent / "../../dependencies/reg2dfa").as_posix()

def get_graph(regex, prefix, directory):

    cmd = ["npm", "start", regex, directory, prefix]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   cwd=REGEX_PATH)

    g = nx.nx_pydot.read_dot("%s/%smin.dot" % (directory, prefix))
    return g

def clean_regex(regex):

    regex = regex.strip()
    regex = regex.replace(" ", "")
    regex = regex.replace("\\", "")
    return regex

def verify(args):

    regex1, regex2 = args

    result = None
    err = None
    start_time = time.time()
    temp_dir = tempfile.TemporaryDirectory()

    try:

        regex1 = clean_regex(regex1)
        regex2 = clean_regex(regex2)

        with contextlib.redirect_stdout(None):
            g1 = get_graph(regex1, "regex1-", temp_dir.name)
            g2 = get_graph(regex2, "regex2-", temp_dir.name)

            result = iso.is_isomorphic(
                g1, g2,
                node_match=iso.categorical_node_match("shape", default=None),
                edge_match=iso.categorical_edge_match("label", default=None))
    except Exception as e:

        err = str(e)

    temp_dir.cleanup()
    return result, start_time, err
