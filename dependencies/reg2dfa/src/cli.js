import {dfa2min, exp2nfa, nfa2dfa, obj2dot} from './main';
import fs from 'fs';

const main = (expr, path, prefix="") => {
  try {
    // NFA
    const nfa = exp2nfa(expr)
    fs.writeFileSync(path + '/' + prefix + 'nfa.dot', obj2dot(nfa));

    // DFA
    const dfa = nfa2dfa(nfa);
    fs.writeFileSync(path + '/' + prefix + 'dfa.dot', obj2dot(dfa));

    // MIN
    const min = dfa2min(dfa);
    fs.writeFileSync(path + '/' + prefix + 'min.dot', obj2dot(min));
  } catch (e) {
    console.error(e.message);
    console.error(e)
  }
};

// Expecting one command line parameter that feeds the 
// regular expression into the main procedure
// console.log(process.argv.slice(2))
main(...process.argv.slice(2));
