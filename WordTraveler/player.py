import random
import logging
from string import ascii_lowercase as letters
from typing import List, Dict, Union

from clemcore.clemgame import Player
from clemcore.backends import Model


# initialize logging:
logger = logging.getLogger(__name__)

# For our game, we need two classes of players.
# First a traveler that will give indications,
# and second a local that will use those indications to navigate the map.


class Traveler(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = [
            """
Arrow 1: West
Clues: luminous
Intended destination: The Statue of Liberty

Arrow 2: South
Clues: desirable, nice
Intended destination: Cannoli
"""]
        
    def _custom_response(self, prompt):
        """
        Args: 
            The prompt to which the player should respond to.
        Returns: 
            Indications with arrow and clues.
        
        """
        indications = self._custom_response(self, prompt)
        return f'{indications}'
        
        
class Local(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = [
"""
Point 1: (1, 2) The Statue of Liberty
Point 2: (3, 4) Cannoli
"""
        ]
        
        def _custom_response(self, prompt):
            target_locations = self._custom_response(self, prompt)
            return f'{target_locations}'
        
        

def check_clue():
    pass

def check_target_locations():
    pass
