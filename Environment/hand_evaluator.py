import math

class HandEvaluator:
    def __init__(self, f_path="../data/hand_rankings.csv"):
        self.hand_rankings = {}
        self.num_cards_in_higher_rank = {}
        self.__create_hand_ranking_and_num_cards_in_higher_rank_hm(f_path)
        self.val_to_num = {
            "2": 2,
            "3": 3,
            "4": 5,
            "5": 7,
            "6": 11,
            "7": 13,
            "8": 17,
            "9": 19,
            "T": 23,
            "J": 29,
            "Q": 31,
            "K": 37,
            "A": 41
        }

        self.suit_to_num = {
            "S": 0b0001 << 27,
            "H": 0b0010 << 27,
            "D": 0b0100 << 27,
            "C": 0b1000 << 27
        }

    def __create_hand_ranking_and_num_cards_in_higher_rank_hm(self, f_path):
        with open(f_path, 'r') as f:
            lines = f.readlines()
            rank = 1
            for line in lines:
                vals = line.strip().split(",")

                card_to_num_val = int(vals[0])
                cards = vals[1]
                flush = int(vals[2])
                num_cards_in_higher_rank_val = int(vals[3])

                key = (card_to_num_val,flush)

                self.hand_rankings[key] = rank
                self.num_cards_in_higher_rank[key] = num_cards_in_higher_rank_val

                rank+= 1

    def check_flush(self, hand):
        return len(set(map(lambda x: x.suit, hand))) == 1

    def hand_to_num(self, hand):
        num = 1
        suits = 0
        for card in hand:
            num *= self.val_to_num[card.value]
            suits |= self.suit_to_num[card.suit]

        return num | suits

    def get_hand_ranking(self, hand):
        hand_val = self.hand_to_num(hand)
        hand_val_without_suits = hand_val & 0b0000111111111111111111111111111
        is_flush = 1 if self.check_flush(hand) else 0
        return self.hand_rankings[(hand_val_without_suits, is_flush)]

    def get_hand_probability_of_winning(self, hand):
        hand_val = self.hand_to_num(hand)
        hand_val_without_suits = hand_val & 0b0000111111111111111111111111111
        is_flush = 1 if self.check_flush(hand) else 0
        _52_chose_5 = math.comb(52, 5)
        prob_winning = (_52_chose_5 - self.num_cards_in_higher_rank[(hand_val_without_suits, is_flush)]) / _52_chose_5
        return prob_winning

    def print_hand_rankings(self):
        for k,v in self.hand_rankings.items():
            print(k,v)

if __name__ == "__main__":
    from DeckOfCards import Card
    evaluator = HandEvaluator()
    hand = [Card("S", "J"), Card("S", "T"), Card("S", "Q"), Card("S", "K"), Card("S", "A")]
    print(evaluator.get_hand_ranking(hand))