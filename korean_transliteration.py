import pickle
import nltk
# nltk.download('cmudict')
from nltk import Tree
from nltk.corpus import cmudict
from hangul_utils import join_jamos
from itertools import product
from collections import defaultdict

# Helper functions

def get_combinations(input_list):
    return list(map(list, product(*input_list)))

def get_ordered_hangul(input_list):
    ordered_hangul = ""
    for _, hangul in input_list:
        ordered_hangul += hangul
    return ordered_hangul

def map_sound(sound, dictionary):
    return [(sound, hangul) for hangul in dictionary[sound]]

def combine_syllables(syllables1, syllables2):
    combined_syllables = []
    for syllable1 in syllables1:
        for syllable2 in syllables2:
            combined_syllable = syllable1 + syllable2
            combined_syllables.append(combined_syllable)
    return combined_syllables

def remove_stress(pronunciations):
    """
    Removes stress levels (numbers) from the pronunciations.

    Example:
        [['W', 'AY1', 'N']] -> [['W', 'AY', 'N']]
    """
    modified_pronunciations = []

    for pronunciation in pronunciations:
        modified_pronunciation = []

        for symbol in pronunciation:
            if symbol[-1].isdigit():
                modified_pronunciation.append(symbol[:-1])
            else:
                modified_pronunciation.append(symbol)

        modified_pronunciations.append(modified_pronunciation)

    return modified_pronunciations

# Grammar and Parser definitions

cvc_grammar = nltk.CFG.fromstring("""
    S -> Syllable | Syllable S
    Syllable -> CVC | CV | CVV | VC | cvT | cvK | cvS | cvR | cvP | cvN | onlyC | onlyV
    cvR -> C V "R"
    cvS -> C V "S"
    cvK -> C V "K"
    cvT -> C V "T"
    cvP -> C V "P"
    cvN -> C V "N"
    CVC -> C V C
    CV  -> C V
    CVV -> C V V
    VC  -> V C
    onlyV -> V
    onlyC -> aloneC
    C -> "K" "S" | "K" | "F" | "S" "H" | "B" "R" | "N" "G" | "P" | "T" "S" | "H" "H" | 'B' | 'M' | 'G' | 'N' 'D' | "C" "H" | "R" | "T" | "V" | "S" | "N" | "D" "R" | "J" "H" | "D" | "L" | 'K' 'R' | 'K' 'L' | "Z" | "G" "R" | "P" "R" | "T" "R" | "W" | "Y" "O" | "Y" "A" | "Y" "A" "A" | "Z" "H"| "R" "D" | "B" "L" | "N" "T"
    aloneC -> "K" | "S" | "Z" | "C" "H" | "V" | "J" "H" | "Y" "O" "W" | "Y" "A" | "F" | "R" "D" | "C" "H" "A" "H" "L"
    V -> "A" "A" | "I" "Y" | "A" "H" | "A" "O" | "A" "E" | 'E' 'H' | "O" "W" | "E" "Y" | "U" "H" | "I" "H" | "Y" "U" "W" | "E" | "A" "Y" | "E" "R" | "U" "W" | "A" "W" | "Y" "O" "W" | "A" "Y" "R" | "W" "A" "Y" | "W" "E" "Y" | "W" "I" "H" | "Y" "U" "H" | "W" "E" | "E" "H" "R" | "O" "Y"
""")

vowel_sound_mapping = {
    'AA': ['ㅏ', 'ㅛ', 'ㅗ'],
    'IY': ['ㅣ'],
    'AH': ['ㅏ', 'ㅓ', 'ㅡ', 'ㅣ', 'ㅔ', 'ㅕ', 'ㅗ', 'ㅜ', 'ㅔ'], # lots of variations here #Joonhyuk, added 'ㅔ'
    'AO': ['ㅓ', 'ㅗ'],
    'AE': ['ㅐ', 'ㅏ'],
    'EH': ['ㅔ', 'ㅏ'],
    'E':  ['ㅔ', 'ㅏ', 'ㅓ'],
    'OW': ['ㅗ'],
    "UH": ['ㅜ', "ㅜㅇㅓ"],
    'IH': ['ㅣ', 'ㅔ'],
    'UW': ['ㅜ', 'ㅡ', 'ㅠ'],
    'EY': ['ㅔㅇㅣ', 'ㅏ'],
    'AY': ['ㅏㅇㅣ'],
    "ER": ['ㅓ', 'ㅓㄹ'],
    "YUW":['ㅠ'],
    "OY": ['ㅗㅇㅣ'],
    #"WAY":['ㅘㅇㅣ'], #instead, add W in cons
    #From here, Joonhyuk added
    "AW": ['ㅏㅇㅜ'],
    'AYR': ['ㅏㅇㅣㅇㅓ'],
    "YOW": ['ㅛ'],
    "WAY": ["ㅘㅇㅣ"],
    "WEY": ["ㅞㅇㅣ"],
    "WIH": ["ㅟ"],
    "YUH": ["ㅠ"],
    "WE": ["ㅝ"],
    "EHR": ["ㅔㅇㅓ"]
}

cons_sound_mapping = {
    'HH': ['ㅎ'],
    'K' : ['ㅋ', 'ㅋㅡ', 'ㄱ'], #Joonhyuk Added 'ㅋㅡ', 'ㄱ'
    'F' : ['ㅍ'],
    'S' : ['ㅅ', 'ㅅㅡ'], #Joonhyuk added 'ㅅㅡ'
    'BR': ['ㅂㅡㄹ'],
    'N' : ['ㄴ', 'ㅇ'],
    'P' : ['ㅍ'],
    'TS': ['ㅈ'],
    'B' : ['ㅂ'],
    'G' : ['ㄱ'],
    'M' : ['ㅁ'],
    'ND': ['ㄴㄷㅡ'],
    'CH': ['ㅊ', 'ㅊㅣ'],
    'R' : ['ㄹ'],
    "T" : ['ㅌ'],
    "V" : ['ㅂ', 'ㅂㅡ'], #Joonhyuk added 'ㅂㅡ'
    "SH": ['ㅅ'],
    'NG': ['ㅇ'],
    'D' : ['ㄷ', 'ㄷㅡ'],
    'JH': ['ㅈ', 'ㅈㅣ'], #Joonhyuk added 'ㅈㅣ'
    "DR": ['ㄷㅡㄹ'],
    "KS": ['ㄱㅅ'],
    "L" : ['ㄹㄹ', 'ㄹ'], # fix this
    "KL": ['ㅋㅗㄹㄹ'],
    "KR": ['ㅋㅗㄹ', 'ㅋㅡㄹ'], #Joonhyuk added "ㅋㅡㄹ'
    "Z" : ['ㅈ', 'ㅅㅡ'],
    "NT": ['ㄴㅌㅡ'],
    #From here, Joonhyuk
    "GR": ['ㄱㅡㄹ'],
    "PR": ['ㅍㅡㄹ'],
    "TR": ['ㅌㅡㄹ'],
    "W": ['ㅇ', 'ㅇㅗ', 'ㅇㅜ'],
    "YOW": ['ㅇㅛ'],
    "YA": ['ㅇㅑ', 'ㅇㅛ'],
    "YAA": ['ㅇㅛ'],
    "ZH": ['ㅈ', 'ㅈㅣ'],
    "RD": ['ㄹㄷㅡ'],
    "BL": ['ㅂㅡㄹㄹ']
}

only_cons_sound_mapping = {
    'K': ['ㅋㅡ'],
    'S': ['ㅅㅡ'],
    'Z': ['ㅈㅡ'],
    'CH': ['ㅊㅣ'],
    'V': ['ㅂㅡ'],
    'JH': ['ㅈㅣ'],
    'YOW': ['ㅇㅛ'],
    'YA': ['ㅇㅑ', 'ㅇㅛ'],
    'F': ['ㅍㅡ'],
    "RD": ['ㄷㅡ'],
    "CHAHL": ['ㅅㅕㄹ']
}

def node_visitor(node: Tree):
    # Visit different types of nodes in the parse tree and perform corresponding actions
    if node.label() == 'C':
        if len(node) == 1:
            return map_sound(node[0], cons_sound_mapping)
        # if consonant is a combination of letters
        cons = ""
        for i in range(len(node)):
            cons += node[i]
        return map_sound(cons, cons_sound_mapping)

    if node.label() == 'aloneC':
        if len(node) == 1:
            return map_sound(node[0], only_cons_sound_mapping)
        # if consonant is a combination of letters
        cons = ""
        for i in range(len(node)):
            cons += node[i]
        return map_sound(cons, only_cons_sound_mapping)

    elif node.label() == 'V':
        if len(node) == 1:
            return map_sound(node[0], vowel_sound_mapping)
        # if vowel is a combination of letters
        vowel = ""
        for i in range(len(node)):
            vowel += node[i]
        return map_sound(vowel, vowel_sound_mapping)

    elif node.label() == 'onlyV' or node.label() == 'VC':
        str_list = [[('?1', 'ㅇ')]] + [node_visitor(n) for n in node]  # Create a list of sound mappings with 'ㅇ' as the first character
        # returns a list of syllables, which are lists of tuples
        return get_combinations(str_list)

    elif node.label() == 'CV' or node.label() == "CVC" or node.label() == 'onlyC':
        # Generate all possible combinations of sound mappings for each sub-node recursively
        return get_combinations([node_visitor(n) for n in node])

    elif node.label() == 'CVV':
        # Create a list of sound mappings with 'ㅇ' in the middle
        str_list = [node_visitor(n) for n in node[:2]] + [[('?', 'ㅇ')]] + [node_visitor(node[-1])]
        return get_combinations(str_list) 

    elif node.label() == 'cvT':
        chars = [node_visitor(n) for n in node[:-1]]   
        chars.append([('T', 'ㅅ'), ('T', 'ㅌㅡ')])  # Append the sound mappings for 'ㅅ' and 'ㅌㅡ'
        return get_combinations(chars)  

    elif node.label() == 'cvP':
        chars = [node_visitor(n) for n in node[:-1]]   
        chars.append([('P', 'ㅂ'), ('P', 'ㅍㅡ')])  # Append the sound mapping for 'ㅂ'
        return get_combinations(chars)

    elif node.label() == 'cvN':
        chars = [node_visitor(n) for n in node[:-1]]
        chars.append([('N', 'ㄴ')])  # Append the sound mapping for 'ㄴ'
        return get_combinations(chars)

    elif node.label() == 'cvR':
        chars = [node_visitor(n) for n in node[:-1]]   
        # shouldn't append anything here
        return get_combinations(chars)

    elif node.label() == 'cvK':
        chars = [node_visitor(n) for n in node[:-1]]   
        chars.append([('K', 'ㅋㅡ')])  # Append the sound mapping for 'ㅋㅡ'
        return get_combinations(chars)  

    elif node.label() == 'cvS':
        chars = [node_visitor(n) for n in node[:-1]]   
        chars.append([('S', 'ㅅㅡ')])  # Append the sound mapping for 'ㅅㅡ'
        return get_combinations(chars)  

    elif node.label() == "Syllable":
        return node_visitor(node[0])  # Join the sound mappings and return as a list

    # If the node represents a combination of multiple syllables
    elif node.label() == "S":
        if len(node) == 1:
            word = node_visitor(node[0])
            return word
        
        return combine_syllables(node_visitor(node[0]), node_visitor(node[1]))

    # If the node does not match any of the above labels, check if it represents a consonant or vowel sound
    else:
        if node[0] in cons_sound_mapping:
            return map_sound(node[0], cons_sound_mapping)
        elif node[0] in vowel_sound_mapping:
            return map_sound(node[0], vowel_sound_mapping)


parser = nltk.ChartParser(cvc_grammar)

sound_frequencies = defaultdict(int)
correct_sound_frequencies = defaultdict(lambda: defaultdict(int))
    

def main():

# Open the file 'eng_to_kor.txt' for reading, assuming it contains English to Korean word mappings
    with open('eng_to_kor.txt', 'r', encoding="utf8") as file:
        i = 0  
        tests = 0  
        tests_passed = 0  

        for line in file:
            word, _, kor_word = line.split()

            # Get the pronunciations of the English word from cmudict dictionary
            pronunciations = cmudict.dict()[word.lower()]

            # Modify the pronunciations by removing stress
            modified_pronunciations = remove_stress(pronunciations)

            tests += 1 
            i += 1  

            test_passed = False  

            for pronunciation in modified_pronunciations:
                input_expression = ''.join(pronunciation)

                try:
                    # Parse the input expression and generate parse trees
                    trees = list(parser.parse(input_expression))
                    
                    # Extract results from the parse trees using a node visitor
                    results = [node_visitor(tree) for tree in trees]
                    
                    hangul_word = ''
                    for result in results:
                        for alt in result:
                            if test_passed:
                                break
                            hangul_chars = get_ordered_hangul(alt)
                            hangul_word = join_jamos(hangul_chars)

                            if hangul_word == kor_word:
                                tests_passed += 1
                                test_passed = True 

                                for sound, hangul_char in alt:
                                    possible_sounds = []
                                    if sound in cons_sound_mapping:
                                        possible_sounds = cons_sound_mapping[sound]
                                    elif sound in vowel_sound_mapping:
                                        possible_sounds = vowel_sound_mapping[sound]
                                    elif sound in only_cons_sound_mapping:
                                        possible_sounds = only_cons_sound_mapping[sound]
                                    if len(possible_sounds) > 1:
                                        correct_sound_frequencies[sound][hangul_char] += 1
                                        sound_frequencies[sound] += 1

                except Exception as e:
                    pass

        print(f'{tests_passed} | {tests}')

        probabilities = defaultdict(dict)

        for sound, frequencies in correct_sound_frequencies.items():
            total_frequency = sound_frequencies.get(sound, 0)
            for character, frequency in frequencies.items():
                probabilities[sound][character] = frequency / total_frequency

        # Check and normalize probabilities if necessary
        for sound, character_probs in probabilities.items():
            total_prob = sum(character_probs.values())
            if total_prob != 1:
                normalization_factor = 1 / total_prob
                for character in character_probs:
                    probabilities[sound][character] *= normalization_factor

        with open('probabilities.pkl', 'wb') as file:
            # Serialize and write the dictionaries to the file
            pickle.dump(probabilities, file)

def predict_Hangul(word, probabilities):
    # Get the pronunciations of the English word from cmudict dictionary
    pronunciations = cmudict.dict()[word.lower()]

    # Modify the pronunciations by removing stress
    modified_pronunciations = remove_stress(pronunciations)

    for pronunciation in modified_pronunciations:
        input_expression = ''.join(pronunciation)

        # try:
        # Parse the input expression and generate parse trees
        trees = list(parser.parse(input_expression))

        # Extract results from the parse trees using a node visitor
        results = [node_visitor(tree) for tree in trees]
        hangul_word = ''
        max_prob = 0.0
        best_option = ''
        for result in results:
            for alt in result:
                current_prob = 1.0
                for sound, hangul_char in alt:
                    if sound in probabilities.keys():
                        if hangul_char in probabilities[sound]:
                            prob = probabilities[sound][hangul_char]
                            current_prob *= prob
                if current_prob > max_prob:
                    max_prob = current_prob
                    best_option = alt
        hangul_chars = get_ordered_hangul(best_option)
        hangul_word = join_jamos(hangul_chars)
        return hangul_word
        
if __name__ == "__main__":
    main()      

    # When probabilities are already saved
    with open('probabilities.pkl', 'rb') as prob_file:
        
        probabilities = pickle.load(prob_file)
        print(probabilities)
        with open('eng_to_kor.txt', 'r', encoding="utf8") as file:
            tests = 0  
            tests_passed = 0  

            for line in file:
                word, _, kor_word = line.split()
                tests += 1
                prediction = predict_Hangul(word, probabilities)
                if not prediction:
                    print(f'Word {word} could not be predicted')
                
                elif prediction == kor_word:
                    tests_passed += 1
                else:
                    print(f'Word {word}')
                    print(f'Prediction {prediction}')
                    print(f'Actual Word {kor_word}')

            print(f'{tests_passed} | {tests}')
