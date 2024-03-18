import re

def replace_rulebase(sentence):
    
    replace_pattern = re.compile(r'\b(?:ristek fasilkom ui|ristek fasilkom|ristek)\b', re.IGNORECASE)
    replaced_sentence = re.sub(replace_pattern, 'RISTEK', sentence)

    remove_pattern = re.compile(r'\b(?:fakultas ilmu komputer universitas indonesia|fakultas ilmu komputer ui|fakultas ilmu komputer|fasilkom ui|fasilkom)\b', re.IGNORECASE)
    replaced_sentence = re.sub(remove_pattern, 'Fasilkom UI', replaced_sentence)
    return replaced_sentence
