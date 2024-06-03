import numpy as np
import pyttsx3

class GameLogic:
    def __init__(self, ruleset, player_names = ['Player 1', 'Player 2'], x01=501, num_legs=1, call_scores=True):
        self.ruleset = ruleset
        self.x01 = x01
        self.num_legs = num_legs
        self.player_names = player_names
        self.num_players = len(self.player_names)
        self.leg_scores = [0] * self.num_players
        self.scores = [x01] * self.num_players
        self.starting_player, self.current_player = 0, 0
        #self.point_history = {i: [[] for _ in range(self.num_legs*2 -1)] for i in range(self.num_players)}
        self.num_dart_history = np.zeros((self.num_players, self.num_legs*2 - 1))
        self.num_visits_history = np.zeros((self.num_players, self.num_legs*2 - 1))
        self.averages = np.zeros(self.num_players)

        self.call_scores = call_scores
        if self.call_scores:
            self.text_to_speech = pyttsx3.init()

    def read_score(self, score):
        if self.call_scores:
            self.text_to_speech.say(str(score))
            self.text_to_speech.runAndWait()

    def get_score_for_dart(self, dart):
        if dart == 'DB':
            return 50
        elif dart == 'SB':
            return 25
        elif dart == 'miss':
            return 0
        else:
            number = int(dart[1:])
            if dart[0] == 'S':
                return number
            elif dart[0] == 'T':
                return number*3
            elif dart[0] == 'D':
                return number*2
            

    def commit_score(self, darts):
        if darts == 'q':
            exit(0)
        if type(darts) == str:
            darts = darts.split()
        
        points = 0
        for dart in darts:
            if (dart[0] not in ['S', 'T', 'D'] or dart[1:] not in [str(x) for x in range(1, 21)]) and dart not in ['SB', 'DB', 'miss']:
                print(f"Invalid dart: {dart}")
                return
            points += self.get_score_for_dart(dart)
        
        self.num_visits_history[self.current_player][np.sum(self.leg_scores)] += 1
        
        self.scores[self.current_player] -= points
        
        if self.ruleset == 'x01':
            self.do_checks_x01_rules(darts, points)
        elif self.ruleset == '121':
            self.do_checks_121_rules(darts, points)
        
    
    def do_checks_x01_rules(self, darts, points):
        num_visits = self.num_visits_history[self.current_player][np.sum(self.leg_scores)]
        
        if self.scores[self.current_player] == 0 and darts[-1][0] == 'D': # check out
            self.num_dart_history[self.current_player][np.sum(self.leg_scores)] += len(darts)
            self.averages[self.current_player] = ((self.averages[self.current_player] * num_visits-1) / num_visits) + ((points * 3/len(darts))/num_visits) 
            self.leg_scores[self.current_player] += 1
            self.scores = [self.x01] * self.num_players
            self.starting_player = (self.starting_player + 1) % self.num_players
            if max(self.leg_scores) == self.num_legs:
                if self.call_scores:
                    self.text_to_speech.say("Game shot, and the match.")
                    self.text_to_speech.runAndWait()
                exit(0)
            else:
                self.current_player = self.starting_player
                if self.call_scores:
                    self.text_to_speech.say(f"Game shot. {self.player_names[self.starting_player]} to throw in leg {sum(self.leg_scores)+1}.")
                    self.text_to_speech.runAndWait()

        else:
            if self.scores[self.current_player] <= 1: # bust
                self.scores[self.current_player] += points # Revert the points
                points = 0 # for average calculation

            self.num_dart_history[self.current_player][np.sum(self.leg_scores)] += 3 # any visit that isn't a checkout is 3 darts
            self.averages[self.current_player] = ((self.averages[self.current_player] * (num_visits-1)) / num_visits) + (points/num_visits)
            self.current_player = (self.current_player + 1) % self.num_players


    def do_checks_121_rules(self, darts, points):
        if self.scores[self.current_player] == 0 and darts[-1][0] == 'D': # check out
            self.leg_scores[self.current_player] += 1
            self.x01 += 1
            self.scores = [self.x01] * self.num_players
            self.starting_player = (self.starting_player + 1) % self.num_players

        elif self.scores[self.current_player] <= 1: # bust
            self.scores[self.current_player] += points # Revert the points
            self.current_player = (self.current_player + 1) % self.num_players

        else: # normal turn
            self.current_player = (self.current_player + 1) % self.num_players
        
        if len(self.point_history[self.current_player][np.sum(self.leg_scores)]) == 3: # this is 3rd visit
            self.scores = [121] * self.num_players


    def play(self):
        
        while True:
            darts = input(f"{self.player_names[self.current_player]}, enter your darts: ")
            self.commit_score(darts)

if __name__ == "__main__":
    game = GameLogic('x01', ['Ben', 'Will'], 301, 2)
    game.play()