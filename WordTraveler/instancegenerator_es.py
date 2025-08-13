"""Template script for instance generation.

usage:
python3 instancegenerator.py
Creates instance.json file in ./in

"""
import json
import os
import random
import logging
import argparse
import string

from clemcore.clemgame import GameInstanceGenerator

GAME_NAME = "word_traveler"
N_INSTANCES = 10 # number of instances per experiment 
N_WORDS = 10
ARROW_PREFIX = "Dirección"
CLUES_PREFIX = "Pistas"
INTENDED_DESTINATION_PREFIX = "Destino"
CARDINAL_DIRECTIONS = "Norte", "Sur", "Este", "Oeste"
NEGATION_PREFIX = "no"
POINT_PREFIX = "Locación"
START_LOCATION = 1
N_TARGET_LOCATIONS = 6
N_TARGET_SPECIAL_LOCATIONS = 3
N_DIMENSIONS = "5x5"  # number of dimensions in the city map
DIMENSION = 5
LANGUAGE = "es"  # language of the game, english
VERSION = "v1.0"  # version of the game

logger = logging.getLogger(__name__) # Set up logging
# Seed for reproducibility
random.seed(73128361) #if you want a different set of experiments, change the seed
# Set up OpenAI API key if using openai
OPENAI_API_KEY = ""  # Insert OpenAI API key


class WordTravelerInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))


    def on_generate(self, *args, **kwargs):
        """Generate game instances for the Word Traveler game."""
        # Set the filename for the generated instances
        self.filename = f"instances_{VERSION}_{LANGUAGE}.json"
        
        numbers= ['one', 'two', 'three']
        experiment_names = []
        difficulty_type = ['strict','nonstrict']
        for difficulty in difficulty_type:
            for x in numbers:
                experiment_names.append(f"{x}_clues_{difficulty}_{LANGUAGE}")
        # for now we will try only with one word list, like in the original game
        experiment = {name: self.add_experiment(name) for name in experiment_names}
    


        for exp_name, experiment in experiment.items():   # build n instances for each experiment
            for game_id in range(N_INSTANCES):

                """ city """
                # load city names and maps, which will be a dictionary with city names as keys and maps as values that will be a nested dictionary
                # load the cities from json file
                with open('resources_es/maps_5x5_es.json', 'r', encoding="utf-8") as f:
                    cities = json.load(f)
                # convert string keys to tuple keys because the maps are stored as tuples in the json file 
                # and when we load them they are converted to strings
                # so we need to convert them back to tuples for consistency
                def make_tuple_keys(cities):
                    """Recursively convert all string keys in a dict to tuples."""
                    if isinstance(cities, dict):
                        return {tuple(map(int, k.split(','))): make_tuple_keys(v) for k, v in cities.items()}
                    elif isinstance(cities, list):
                        return [make_tuple_keys(i) for i in cities]
                    else:
                        return cities
                # sample a random city from the list of cities
                city_name = random.choice(list(cities.keys()))
                # get the map for the city, it will be the value of the city_name key
                city_map = cities[city_name]
                city_map = self.tuple_keys_to_str(city_map)

                """ map """
                # load locations from the city map to include coordinates and names
                locations = list(city_map.items())
                starting_location = random.sample(locations, START_LOCATION)
                target_locations = random.sample(locations, N_TARGET_LOCATIONS)
                remaining_locations = [loc for loc in locations if loc not in target_locations]
                special_target_locations = random.sample(remaining_locations, N_TARGET_SPECIAL_LOCATIONS)

                # turn the city map into a more readable format.
                string_city_map = "\n".join(f"{k}: {v}" for k, v in city_map.items())
                string_starting_location = "\n".join(f"{coord}: {name}" for coord, name in starting_location)
                string_target_location = "\n".join(f"{coord}: {name}" for coord, name in target_locations)
                string_special_target_locations = "\n".join(f"{coord}: {name}" for coord, name in special_target_locations)

                """ number of arrows """
                arrow_dict = {
                    '3': ['three_clues_strict_es', 'three_clues_nonstrict_es'],
                    '2': ['two_clues_strict_es', 'two_clues_nonstrict_es'],
                    '1': ['one_clues_strict_es', 'one_clues_nonstrict_es']
                }
                # determine the number of arrows based on the experiment name
                n_arrows = next((k for k, v in arrow_dict.items() if exp_name in v), '0')
                
                """ difficulty_type """
                
                difficulty_dic = {
                    'nonstrict':['three_clues_nonstrict_es', 'two_clues_nonstrict_es', 'one_clues_nonstrict_es'],
                    'strict': ['three_clues_strict_es', 'two_clues_strict_es', 'one_clues_strict_es'],
                }
                # determine the difficulty type based on the experiment name
                difficulty = next((k for k, v in difficulty_dic.items() if exp_name in v), '0')
                
                """ word lists """
                adjectives = self.load_file('resources_es/wordlist_es.txt').strip('\n').split('\n')
                word_list = random.sample(adjectives, N_WORDS) # default word list for the game from whcih we will sample 10 words

                '''populate the instance with its parameters'''
                instance = self.add_game_instance(experiment, game_id)
                instance['n_dimensions'] = DIMENSION
                instance['city'] = city_name
                instance['city_map'] = city_map
                instance['word_list'] = word_list
                instance['starting_location'] = starting_location
                instance['target_locations'] = target_locations
                instance['special_target_locations'] = special_target_locations
                instance['arrow_prefix'] = ARROW_PREFIX
                instance['n_arrows'] = n_arrows
                instance['clues_prefix'] = CLUES_PREFIX
                instance['intended_destination_prefix'] = INTENDED_DESTINATION_PREFIX
                instance['cardinal_directions'] = CARDINAL_DIRECTIONS
                instance['negation_prefix'] = NEGATION_PREFIX
                instance['point_prefix'] = POINT_PREFIX
                instance['difficulty'] = difficulty  
                instance['prompt_player_a'] = self.create_prompt(self.load_template('resources_es/prompts/traveler_prompt_es.template'), N_DIMENSIONS, ARROW_PREFIX, CLUES_PREFIX, INTENDED_DESTINATION_PREFIX, CARDINAL_DIRECTIONS, NEGATION_PREFIX, None, city_name, string_city_map, n_arrows, string_starting_location, string_target_location, string_special_target_locations, word_list)
                instance['prompt_player_b'] = self.create_prompt(self.load_template('resources_es/prompts/local_prompt_es.template'), None, None, None, None, None, None,POINT_PREFIX, city_name, string_city_map, None, string_starting_location, None, None,None)
                errors_file = os.path.join('resources_es', 'errors_es.json')
                with open(errors_file, 'r', encoding='utf-8') as file:
                    instance['error_messages'] = json.load(file)                

    def tuple_keys_to_str(self, d):
        """Recursively convert all tuple keys in a dict to strings."""
        if isinstance(d, dict):
            return {str(k): self.tuple_keys_to_str(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.tuple_keys_to_str(i) for i in d]
        else:
            return d
    
    def create_prompt(self,
                      prompt: str,
                      n_dimensions: str = N_DIMENSIONS,
                      arrow_prefix: str = ARROW_PREFIX,
                      clues_prefix: str = CLUES_PREFIX,
                      intended_destination_prefix: str = INTENDED_DESTINATION_PREFIX,
                      cardinal_directions: list = CARDINAL_DIRECTIONS,
                      negation_prefix: str = NEGATION_PREFIX,
                      point_prefix: str = POINT_PREFIX,
                      city_name: str = None,
                      city_map: str = None,
                      n_arrows: str = None,
                      starting_location: str = None,
                      target_locations: list = None,
                      special_target_locations: list = None,
                      word_list: list = None) -> str:
        """Replace a prompt template with slot values."""
        text = string.Template(prompt).substitute(
            n_dimensions=n_dimensions,
            arrow_prefix=arrow_prefix,
            clues_prefix=clues_prefix,
            intended_destination_prefix=intended_destination_prefix,
            cardinal_directions=cardinal_directions,
            negation_prefix=negation_prefix,
            point_prefix=point_prefix,
            city_name=city_name,
            city_map=city_map,
            n_arrows=n_arrows,
            starting_location=starting_location,
            target_locations=target_locations,
            special_target_locations=special_target_locations,
            word_list=word_list,
        )
        return text



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate game instances for Word Traveler.")
    parser.add_argument("-m", "--mode", choices=["manual"], default="manual",
                        help="Mode of instance generation. Currently only 'manual' is supported.")
    args = parser.parse_args()
    WordTravelerInstanceGenerator().generate(filename=f"instances_{LANGUAGE}.json")