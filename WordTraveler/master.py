import os.path
from typing import Dict, Tuple, List, Union
import logging
import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemcore.clemgame.master import ParseError, RuleViolationError, GameError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from clemcore.utils import string_utils


from sentence_transformers import SentenceTransformer, util 

logger = logging.getLogger(__name__)

from pprint import pprint
# For the parsing.
import re



class Traveler(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = [
"""
Direction: West
Clues: luminous
Destination: The Statue of Liberty

Direction: South
Clues: desirable, nice
Destination: Cannoli
"""
            ]
        
    def _custom_response(self, prompt) -> str:
        """
        Args: 
            The prompt to which the player should respond to.
        Returns: 
            Indications with direction and clues.
        """
        response = self._custom_responses[np.random.randint(0, len(self._custom_responses))]
        return response
        
        
class Local(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = [
"""Location 1: (1, 2) The Statue of Liberty
Location 2: (3, 4) Cannoli
"""
        ]
        
    def _custom_response(self, prompt) -> str:        
        """
        Args: 
            The prompt to which the player should respond to.
        Returns: 
            Indications with locations. 
        """
        target_locations = self._custom_response(self, prompt)
        return f'{target_locations}'
    
    
# Travelers have to provide 3 indications.
# Moves must be in the vertical or horizontal axis.
# They can only use one of four cardinal directions for the arrows.
# And the adjectives given to them as a word list.
# They can negate the adjectives if needed ("funny" and "not funny" are both valid answers).
# and they can only use each adjective once.

def check_answer(player, parsed_response, word_list, cardinal_directions, starting_location, dimension= None, negation= None, city_map = None, arrows = None, errors = None):

    # The X and Y locations allow us to see weather the llm is actually moving horizontally or vertically, 
    # or just jumping around in the map.
    
    starting_x, starting_y = starting_location[0][0].strip("()").split(",")
    starting_x, starting_y = int(starting_x), int(starting_y)
    
    
    
    if isinstance(player, Traveler):
        
        # For startes, we load a model from SentenceTransformers
        # This will later allow us to see how well the llm maps the clues to the locations through a ranking system.
        # We also turn the adjectives as embeddings already, but from a different list, since we will remove items from this list.
        model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
        second_word_list = word_list
        adjective_ranks = []


        
        # This list will hold all adjectives used
        adjectives_used = []
        
        # First, we make sure the movements of the LLM respect the x and y axis.
        # We check all the answers, we extract the coordinates and register them.
        coord_lookup = { name: coordinates for coordinates, name in city_map.items() }

        x, y = [], []
        for item in parsed_response:
            dest = item["destination"].rstrip('.')
            # Sometimes the answer would come with a dot, causing always a mistake 

            if dest not in coord_lookup:
                reason = errors['location_error']
                raise RuleViolationError(reason, parsed_response)
            coord_str = coord_lookup[dest]
            x_str, y_str = coord_str.strip("()").split(",")
            x.append(int(x_str))
            y.append(int(y_str))
        

        for i in range(len(x)):
            
            """ The error feedback is not as specific in this case. """
            if i == 0:
                if not (x[i] == starting_x) ^ (y[i] == starting_y):
                    reason = errors['coordinates_error']
                    raise RuleViolationError(reason, parsed_response)
            else:
                if not (x[i] == x[i - 1]) ^ (y[i] == y[i - 1]):
                    reason = errors['coordinates_error']
                    raise RuleViolationError(reason, parsed_response)
                
        # Then we go response by response to check for other general parameters.
        for response in parsed_response:
            
            # Check cardinal directions 
            response["direction"] = response["direction"].rstrip('.')
            if response["direction"] not in cardinal_directions:
                reason = errors['cardinal_direction_error']
                raise RuleViolationError(reason, response)
            
            # Check that all clues have at least one adjective: 
            if response["clues"] == None:
                reason = errors['clue_with_no_adjectives']
                raise RuleViolationError(reason, response)
            
            # Check adjectives.
            # The idea is: First, check if there is a negation in languages where a space is used.
            
            # (response["clues"])
            
            for adjective in response["clues"]:
                adjective = adjective.rstrip('.')
                                
                if " " in adjective:
                    words = adjective.split() 
                    
                    # If there is a space in our clues, a modifier has been used.
                    # We have to ensure that the modifier used is the negation, and not another particle or adverb.
                    if negation and negation not in words:
                        reason = errors['not_negation_used']
                        raise RuleViolationError(reason, response)  
                    
                    for word in words:
                        word = word.rstrip('.')
                        # Ignore the negation word.
                        if word == negation:
                            pass
                        else:
                            
                            if not word in word_list:            
                                reason = errors['adjective_not_found']
                                raise RuleViolationError(reason, response) 
                                                
                            if word in adjectives_used:
                                    reason = errors['adjective_used_twice']
                                    raise RuleViolationError(reason, response)   
                            else:
                                
                                # We embbed both the location and the adjectives avaialable.
                                # Then we give them a cosine similarity.
                                emb_noun = model.encode(response["destination"],  convert_to_tensor=True)
                                emb_adjectives = model.encode(second_word_list, convert_to_tensor=True)
                                cos_scores = util.pytorch_cos_sim(emb_noun, emb_adjectives)[0].tolist() 
                                
                                # We create an array with the ranks.
                                # Since this logical branch is for negated adjectives, the scores are inverted.
                                # (Normally, we negate scores_array)
                                scores_array = np.array(cos_scores)
                                descending_idx = np.argsort(scores_array)
                                
                                # And now we assign ranks
                                ranks = np.empty_like(descending_idx)
                                ranks[descending_idx] = np.arange(1, len(scores_array) + 1)
                    
                                # We zip the values and create a dictionary out of them.
                                rank_dict = dict(zip(second_word_list, ranks.tolist()))
                                
                                # We keep track of the rank of the adjective used.
                                adjective_ranks.append(rank_dict[word])

                                # We remove the value from our secondary list so that it is no longer 
                                # taken into for the ranking.
                                second_word_list.remove(word)

                                #We append the adjective to the list of adjectives used.
                                adjectives_used.append(word)
                

                
                # The adjective is a singular word. If that's the case, let's see if it's used in the word list.
                else:       
                    if adjective not in word_list:
                        reason = errors['adjective_not_found']
                        raise RuleViolationError(reason, response)
                    
                    # If they are in the word list now, we need to check if they have already been used.
                    else: 
                        if adjective in adjectives_used:
                            reason = errors['adjective_used_twice']
                            raise RuleViolationError(reason, response)
                        
                        # If they are not in the used adjectives list, they are valid.
                        else:
                            
                            # If the game goes well, we can then compare the noun with the adjective
                            
                                # We embbed both the location and the adjectives avaialable.
                                # Then we give them a cosine similarity.
                                emb_noun = model.encode(response["destination"],  convert_to_tensor=True)
                                emb_adjectives = model.encode(second_word_list, convert_to_tensor=True)
                                cos_scores = util.pytorch_cos_sim(emb_noun, emb_adjectives)[0].tolist() 
                                
                                # We create an array with the scores.
                                scores_array = np.array(cos_scores)
                                descending_idx = np.argsort(-scores_array)
                                
                                # And now we assign ranks
                                ranks = np.empty_like(descending_idx)
                                ranks[descending_idx] = np.arange(1, len(scores_array) + 1)
                    
                                # We zip the values and create a dictionary out of them.
                                rank_dict = dict(zip(second_word_list, ranks.tolist()))
                                
                                # We keep track of the rank of the adjective used.
                                adjective_ranks.append(rank_dict[adjective])

                                # We remove the value from our secondary list so that it is no longer 
                                # taken into for the ranking.
                                second_word_list.remove(adjective)

                                #We append the adjective to the list of adjectives used.
                                adjectives_used.append(adjective)
                            
        # For scoring purposes, we add the avarage quality of the adjectives used.
        return adjective_ranks, sum(adjective_ranks) / len(adjective_ranks)
                            
                               
        
    # The local has a similar logic to the traveler.
    elif isinstance(player, Local):
        # It need to move only in the horizontal or vertical axis.
        for idx, response in enumerate(parsed_response):
            if idx == 0:
                if not (response['x'] == starting_x) ^ (response['y'] == starting_y):
                    reason = errors['coordinates_error']
                    raise RuleViolationError(reason, idx) 
            else:
                if not (response['x'] == parsed_response[idx - 1]['x']) ^ (response['y'] == parsed_response[idx - 1]['y']):
                    reason = errors['coordinates_error']
                    raise RuleViolationError(reason, idx)  

                    
        # It needs to respect the city boundries, 
        if not (0 <= int(response["x"]) <= int(dimension) and 0 <= int(response["y"]) <= int(dimension)):
            reason = errors['coordinates_outside_map']
            raise RuleViolationError(reason, response)
        
        # and go to real locations 
        response["location"] = response["location"].rstrip('.')
        if response["location"] not in city_map.values():
            reason = errors['location_error']
            raise RuleViolationError(reason, response)
        
        
            

      
                     
            

class WordTraveler(DialogueGameMaster):
    """Implement mechanisms for playing Word Traveler, a map navigation game with word guesses,
    in which player A (Traveler) has to describe his target detinations with a limited set of adjectives and movements
    to player B (Local), who has to interpret this clues and move around the map.
    Rule following is checked by the check_answer function. 
    
    """

    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)
        
        self.max_attempts = 4
        self.current_attempts = 1
        self._repeat_turn = False
    
    def _on_setup(self, **game_instance):
        
        self.score = 0
        self.avarage_adj_rank = 0
        self.game_instance = game_instance
        
        "Set a difficulty (strict vs nonstrict)"
        self.difficulty = game_instance["difficulty"]
        
        # Initialize the instance in the second prompt for round 1
        self.city_map = game_instance["city_map"]
        self.target_locations = game_instance["target_locations"]
        self.special_locations = game_instance["special_target_locations"]
        self.word_list = game_instance["word_list"]
        self.dimension = game_instance["n_dimensions"]
        self.arrows = int(game_instance["n_arrows"])
        self.starting_location = game_instance["starting_location"]
        self.negation = game_instance["negation_prefix"]
        
        # Initialize prefixes for the game.
        self.arrow_prefix = game_instance["arrow_prefix"]
        self.clues_prefix = game_instance["clues_prefix"]
        self.cardinal_directions = game_instance["cardinal_directions"]
        self.intended_destination_prefix = game_instance["intended_destination_prefix"]
        self.point_prefix = game_instance["point_prefix"]
        
        # initialize regex patterns and error messages
        self.arrow_pattern = re.compile(rf"{re.escape(self.arrow_prefix)}[-\s]*(?P<arrow_number>\d*)\s*:(?P<direction>.+)", re.IGNORECASE) #arrow number and the direction as named group
        self.clues_pattern = re.compile(rf"{re.escape(self.clues_prefix)}:\s*(?P<clues>.+)", re.IGNORECASE) #clues as a named group.
        self.intended_destination_pattern = re.compile(rf"{re.escape(self.intended_destination_prefix)}:\s*(?P<destination>.+)", re.IGNORECASE) # destination as a named group.
        self.point_pattern = re.compile(rf"{re.escape(self.point_prefix)}[-\s]*(?P<point_number>\d*)\s*:\s*\((?P<coords>\s*\d+\s*,\s*\d+\s*\))\s*(?P<location>.+)", re.IGNORECASE) # point number, coords and location as named groups.
        
        self.error_messages = game_instance["error_messages"]
        
        # For the traveler
        self.prompt_player_a = game_instance["prompt_player_a"]
        
        # For the local
        self.prompt_player_b = game_instance["prompt_player_b"]
        
        # Define the players
        self.traveler = Traveler(self.player_models[0])
        self.local = Local(self.player_models[1])
        
        # Add Players
        self.add_player(self.traveler, initial_context=self.prompt_player_a)
        self.add_player(self.local)
                
        # We set variables that will control weather the game continues or has to be aborted.
        self.success, self.aborted, self.failure = False, False, False
        
        self.game_error = None
        
    # Checks if the game can proceed or not.    
    def _does_game_proceed(self):
        return not (self.aborted or self.failure or self.success)
        
    def _on_game_error(self, error: GameError):
           
        player = self.current_player 
        # Logic for nonstrict games.
        if isinstance(error, RuleViolationError) and self.difficulty == "nonstrict":
            self.current_attempts += 1   
            if self.current_attempts <= self.max_attempts:
                self.set_context_for(
                    player, error.reason
                )
                self.log_to_self(error.reason, error.reason)
                self._repeat_turn = True
                
                
            else:
                self.log_to_self(error.reason, error.reason)
                self.failure = True
                    
        else:
            self.log_to_self(error.reason, error.reason)
            self.failure = True
                
                
    def _should_pass_turn(self):
        # If the repeat_turn self boolean got activated, stay on the same turn.
        if self._repeat_turn == True:
            self._repeat_turn = False
            return False
        else:
            return True
        
    def _advance_game(self, player:Player, parsed_response: list):
        
        results = check_answer(player, parsed_response, self.word_list, self.cardinal_directions, self.starting_location, self.dimension, self.negation, self.city_map, self.arrows, self.error_messages)
        adjective_ranks, average_rank = results if results is not None else ([], 0.0)


        
        self.log_to_self(": valid indications/points", parsed_response)
        
        if player == self.traveler:
            responses_together = []
            for entry in parsed_response:
                # append each response together to replace the prompt later on.
                responses_together.append(entry["response"])
            # Now we combine the responses into a single string.
            combined = "\n".join(responses_together)
            
            # Now we set the context for the local player.
            self.prompt_player_b = self.prompt_player_b.replace("_starting_location", self.starting_location[0][0])
            self.set_context_for(self.local, self.prompt_player_b.replace("_clues", combined))
            
            self.log_to_self("Avarage adjective used rank:", average_rank)
            self.avarage_adj_rank = average_rank
            
            self.log_key('adjective_ranks', adjective_ranks) 
            self.log_to_self('adjective_rank', adjective_ranks) 
            logger.info(adjective_ranks)

            
        # Let's try to validate the local answer now.
        if player == self.local:
        
        # Let's compute the score of this instance and finish the game.
            for idx, response in enumerate(parsed_response): 
                match = False
                
                for sptl in self.special_locations:
                    if response["location"] != sptl[1]:
                        pass
                    else:
                        self.log_to_self(f"guess {idx} right", "success!")
                        self.score += 2
                        match = True
                        break
                
                if not match:
            
                    for tl in self.target_locations:
                    
                        if response["location"] != tl[1]:
                            pass
                        else:
                            self.log_to_self(f"guess {idx} right", "success!")
                            self.score += 1
                            match = True
                            break
                    
                if not match:
                    self.log_to_self(f"guess {idx} wrong", "failure!")
                    
            # self.compute_turn_score()
            self.log_to_self("instance completed", "end game")
            self.success = True
    
        
    def _parse_response(self, player:Player, response:str) -> str:
        #split the response into blocks by \n\n
        #then split the blocks by \n
        #check line by line if it starts with the correct prefix
        
        results = []    
        
        if player == self.traveler: # Traveler's turn
           #now this treats the case that the player doesnt use a blank space between the two blocks
           #we define a block as the ones starting with the arrow prefix (While it keeps the old name, now it's direction rather than arrow)
            lines = response.strip().splitlines()
            response_blocks = []
            current_block = []
        

            for line in lines:
                if line.startswith(self.arrow_prefix) and current_block:
                    response_blocks.append("\n".join(current_block))
                    current_block = [line]
                else:
                    current_block.append(line)
            if current_block:
                response_blocks.append("\n".join(current_block))

            logger.info(f"response blocks: {response_blocks}")
            for block in response_blocks:
            
                logger.info(f"Processing block: {block}")
                block_lines = block.strip().split("\n") # split the block into lines
                logger.info(f"Processing block line: {block_lines[0]}")

                if len(block_lines) < 3:
                    reason = self.error_messages["parse_error"]
                    raise ParseError(reason, response)

                #process arrow here
                arrow_match= self.arrow_pattern.search(block_lines[0])
                if not arrow_match:
                    logger.info(f"Arrow match failed for block: {block_lines[0]}")
                    reason = self.error_messages["parse_error_direction"]
                    raise ParseError(reason, response)
                logger.info(f"Arrow number: {arrow_match.group('arrow_number')}")
                logger.info (f"Direction: {arrow_match.group('direction')}")          
                arrow_num   = arrow_match.group('arrow_number')
                direction   = arrow_match.group('direction').strip()
                
      
                # process clues here
                clues_match = self.clues_pattern.search(block_lines[1]) # match the second line with the clues pattern
                if not clues_match:
                    logger.info(f"Clues match failed for block: {block_lines[0]}")
                    reason = self.error_messages["parse_error_clues"]
                    raise ParseError(reason, response)
                logger.info(f"Clues: {clues_match.group('clues')}")
                clues_text = clues_match.group('clues').strip()
                clues_list = [c.strip() for c in clues_text.split(",") if c.strip()]
                
                # process destination here
                destination_match = self.intended_destination_pattern.search(block_lines[2]) # match the third line with the intended destination pattern
                if not destination_match:
                    logger.info(f"Intended destination match failed for block: {block_lines[0]}")
                    reason = self.error_messages["parse_error_destination"]
                    raise ParseError(reason, response)
                destination = destination_match.group('destination').strip()
                logger.info(f"Destination: {destination_match.group('destination')}")
                
                
                # track results
                results.append({
                "response": rf"{self.arrow_prefix}: {arrow_num}, {direction}. {self.clues_prefix}: {clues_list}",
                "arrow": arrow_num,
                "direction": direction,
                "clues": clues_list,
                "destination": destination
                })

            # Processed all blocks, now we check the results.    
            logger.info(f"Processed response blocks. Blocks_used={len(results)}")
            if len(results) > self.arrows:
                reason = self.error_messages["parse_error_movements_more"]
                raise RuleViolationError(reason, len(results))
            if len(results) < self.arrows:
                reason = self.error_messages["parse_error_movements_less"]
                raise RuleViolationError(reason, len(results))
            return results
               
        elif player == self.local: 
            
            lines = response.strip().splitlines()
            response_blocks = []
            
            current_block = []

            for line in lines:
                if line.startswith(self.point_prefix) and current_block:
                    response_blocks.append("\n".join(current_block))
                    current_block = [line]
                else:
                    current_block.append(line)
                    
            if current_block:
                response_blocks.append("\n".join(current_block))
            
            
            for block in response_blocks:
                logger.info(f"Processing block: {block}")
                # we dont need to split for block lines because one block of the traveler's reasponse is only one line
                #process point here
                point_match = self.point_pattern.search(block)
                if not point_match:
                    logger.info(f"Point match failed for block: {block}")
                    reason = self.error_messages["parse_error_points"]
                    raise ParseError(reason,response)
                logger.info(f"Point number: {point_match.group('point_number')}")
                logger.info(f"Coordinates: {point_match.group('coords')}")
                logger.info(f"Location: {point_match.group('location')}")
   
                point_num = point_match.group('point_number')
                coords    = point_match.group('coords').strip()
                location = point_match.group('location').strip()
                x_str, y_str = re.findall(r"\d+", coords)
                x, y = int(x_str), int(y_str)
                logger.info(f"Point number: {point_num}, Coords: {coords}, Location: {location}")
                
                results.append({
                "response": block,
                "point": point_num,
                "x": x,
                "y": y,
                "coordinates": "(" + coords,
                "location": location
            })
                
            if len(results) > self.arrows:
                reason = self.error_messages["parse_error_movements_more"]
                raise RuleViolationError(reason, len(results))
            if len(results) < self.arrows:
                reason = self.error_messages["parse_error_movements_less"]
                raise RuleViolationError(reason, len(results))
            return results 
            
            
            
    # Currently we are not using this Error function just because RuleviolationError lets us return a more personalized error message.
    def _on_parse_error(self, error: ParseError):
        player = self.current_player 
        if isinstance(error, ParseError) and self.difficulty == "nonstrict":
            self.current_attempts += 1   
            if self.current_attempts <= self.max_attempts:
                self.set_context_for(
                    player, error.reason
                )
                self.log_to_self(error.reason, "warning")
                self._repeat_turn = True
                
                
            else:
                self.log_to_self(error.reason, error.reason)
                self.aborted = True
                    
        else:
            self.log_to_self(error.reason, error.reason)
            self.aborted = True
            
            
    """This is for the whole scoring"""
        
    def compute_turn_score(self):
        return 1 if self.success else 0

    def compute_episode_score(self):
        if self.success:
            return 100 / (self.current_attempts)
        return 0
    
    def _on_after_game(self):
        
        if self.success:
            self.log_key('success', 'true')
        else:
            self.log_key('success', 'false')
            
        
        score = self.score
        arrows = self.arrows
        attempts = self.current_attempts
        rank_avg = self.avarage_adj_rank
                
        # Technically, you can make more points than the number of moves because of special target locations, so we need to account for that.
        if score > arrows:
            score = arrows
        score = (score / arrows) * 100
        speed = 100 / attempts

        self.log_key('arrows', self.arrows) 
        self.log_key('Rank', rank_avg) 
        self.log_key('attempts', self.current_attempts)
        self.log_key('Speed', speed)
        self.log_key('Rank', rank_avg) 
        self.log_key(BENCH_SCORE, self.score)
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_SUCCESS, int(self.success))
        self.log_key(METRIC_LOSE, int(not self.success))
        
        # self.log_to_self("log metric", f"Score: {score}") 
        # self.log_to_self("log metric", f"Speed: {speed}") 
            
        

class WordTravelerScorer(GameScorer):
    # …

    def compute_episode_scores(self, interactions: Dict):
        
        score = interactions[BENCH_SCORE]
        arrows = interactions["arrows"]
        speed = interactions["Speed"]
        rank_avg = interactions["Rank"]
        
        # Technically, you can make more points than the number of moves because of special target locations, so we need to account for that.
        if score > arrows:
            score = arrows
        score = (score / arrows) * 100
        
        self.log_episode_score(BENCH_SCORE, score)
        
        if interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, score)
            self.log_episode_score("Speed", speed)
            self.log_episode_score("Rank", rank_avg)
        elif interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, -1)
            self.log_episode_score("Speed", 0)
            self.log_episode_score("Rank", rank_avg)
        elif interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
            self.log_episode_score("Speed", 0)
            self.log_episode_score("Rank", 0)
        else:
            raise ValueError("Missing outcome value (success, failure, abort) in interactions.json")


class WordTravelerBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return WordTraveler(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return WordTravelerScorer(self.game_name, experiment, game_instance)

