
# Word Traveler

**Implemented by:** Evelina Bourazani & Juan Danza  
**Inspired by:** _Word Traveler_ by Thomas Dagenais-Lespérance (Office Dog, 2024)

**The implementation of this game was done within the scope of Clembench for LLM benchmarking purporses. The purpose of the game is to test and evaluate LLMS' abilities through making them play the game.**

**Word Traveler** is a cooperative two-player word-guessing and map-navigation game. One player is the **Traveler**, giving clues using a limited set of adjectives to indicate target locations on a city map. The other is the **Local**, interpreting the clues to identify those locations.  
Correct guesses earn points. Variations change the number of targets (1–3) and strictness mode (strict vs non-strict).

This version adapts the board game Word Traveler into the _clembench_ framework with text-based maps, command-line interaction, and multi-language support.  The implementation does not fully follow the structure of the original board game, but is rather inspired by it.

## Clembench

Clembench is a framework for the systematic evaluation of chat-optimized Language Models as Conversational Agents.

The core framework code used to run the game is Clemcore. It is recommended to install Clemcore in its own separate Python3.10 virtual environment. The packaged library can be installed using:
```
pip install clemcore
```

## The game

### Special Terms
- **Traveler**: clue giver.
- **Local**: clue guesser.
- **Direction**: referring to cardinal directions (North, South, East, West).
- **Clues**: word clues that are adjectives. According to the clue giver, these adjectives are related to the intended destination.
- **Destination**: the intended destination the clue giver is giving clues for.
- **Set of directions**: one set of directions consists of one (cardinal) direction, (word) clues and one (intended) destination.
- **Location**: the guess provided by the guesser, based on the set of directions formed by the clue giver.
- **Starting location**: the location on the map that will be the starting point of navigation for that round.
- **Target locations**: a set of locations on the map that if visited successfully, grant 1 point to the team.
- **Special target locations**: a set of locations on the map that if visited successfully, grant 2 points to the team.

### Game Description
In this game, the players are in a city and one of them (Traveler) wants to explore the city and describes which locations they want to visit, while the other (Local) must guess which locations would fit the descriptions. 

At the beginning of the game, each player is assigned a distinct role, either 'Traveler' or 'Local'. 
The **Traveler** receives a text-based city map with locations marked by (x, y) coordinates, a list of target and special target locations, and a set of 10 adjectives to use as clues. The Traveler's mission is to give clear directions that guide the Local to the intended destinations.
The **Local** has the same city map and receives the Traveler’s clues. Using these, they must identify which locations the Traveler wants to visit.

In each game, the number of locations to visit is predetermined for each game. A location is considered successfully visited when the Local correctly guesses it based on the Traveler’s directions. The team earns points for each correct guess and successfully visited location.

## Resources

Each language has its own folder containing:
- **Prompts** – Player instructions, one for each player.
- **Errors** – Feedback messages and reprompting, only for non-strict mode.
- **Maps** – City grids (5×5). There are currently 16 city maps. Each map includes real or made-up landmarks and items to represent the 'locations'.
- **Word lists** – 160 adjectives, based on the original board game word list.

## Experiments

There are six different experiments in total, variating with different number of clues to be given and level of strictness.

**NUMBER OF LOCATIONS**: There are three different experiments with variating number of locations the must visit. In each experiment respectively, the players must visit 1, 2 or 3 locations at once.

**STRICTNESS**: There is strict and non-strict version. In strict version, the player only has one chance to send the clues, while in non-strict version, in case there is a mistake in the format of the answer, then feedback is provided to the player and they can try again, up to three times.

The above experiments are combined and create a total of 6 final experiments, with all three location experiments provided in both strict and non-strict mode.

## Languages

Currently supported languages:
- English
- Spanish
- Greek

# RUN

## Instance Generation

One generation file per language, plus a special human-player mode.  

To generate instances, run:

```bash
python3 word_traveler/instancegenerator_en.py
```

(Change filename for other languages or modes.)
## Running the game
To run instances, use the command 'clem run' and specify the game name under the -g flag. Additionally, the -m flag specifies the model(s). If there is one model under it, then it plays against itself. To make a model play against another model, simple add a second model name. When the second model name is 'human', then a human can play against a model, while when both model names are 'human', then two humans can play against eachother. The positon of the model name as first or second indicates the role the player will have in the game. For example, if a human wants to play as the Local against a model, then 'human' should be the second model name, as the Local is Player 2. The -i flag has to be used along with the specific instances file name. Change the file name under the -i flag for other languages/modes/experiments. Finally, use the -l flag usually set to 300, to allow the models for more token generation. This flag is automatically set to 100. 
The following commands show a set of possibilities to run the game with different model players:

```shell
clem run -g WordTraveler -m grok-3 -i instances_en.json -l 300

clem run -g WordTraveler -m mock -i instances_el.json -l 300

clem run -g WordTraveler -m gemma-3-27b human -i instances_en.json -l 300

clem run WordTraveler -m human deepseek-v3 -i instances_es.json -l 300 
```

## Transcribing

To transcribe the model interactions into transcripts in htlm and tex (the flag -g is optional):

```shell
clem transcribe -g WordTraveler
```
## Scoring

To score previous runs (the flag -g is optional):

```bash
clem score -g WordTraveler
```

**Metrics:**

- **Quality** - Number of correct guesses
- **Speed** - `100 / t` where _t_ is the turn of the first correct answer (0 if unsuccessful/aborted)
- **Outcome** - Success, Loss, or Aborted
- **Word rank** - Uses a transformer to produce a cosine similarity score between a location and the available adjectives the players has at their disposal, which are later ranked. 1 indicates the highest similarity possible.

## Evaluation
To create evaluation boards(the flag -g is optional):
```
clem eval -g WordTraveler
```
