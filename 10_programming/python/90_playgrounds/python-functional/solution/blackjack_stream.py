import random
from itertools import product, starmap, cycle
from monads import MultiValue
from utils import compose2, update_dict, stream_logs
from card_games import Card, Deck, Hand, draw_to_hand

BUST_LIMIT = 22

def new_deck() -> Deck:
    return list(starmap(Card, product(range(1, 14), "SHDC")))

def shuffle(deck: Deck) -> Deck:
    return random.sample(deck, len(deck))

new_shuffled_deck = compose2(new_deck, shuffle)

def get_points_for_card(card: Card):
    match card.rank:
        case 10 | 11 | 12 | 13:
            return MultiValue({10})
        case 1:
            return MultiValue({1, 11})
        case _:
            return MultiValue({card.rank})

def get_points(hand: Hand) -> MultiValue:
    return sum((get_points_for_card(card) for card in hand), MultiValue({0}))

def score_hand(hand: Hand):
    total = get_points(hand)
    allowed_values = (value for value in total.values if value < BUST_LIMIT)
    return max(allowed_values)


def get_player_action(hand, logs):
    stream_logs(logs)
    while (action := input('(H)it or (S)tay: ').lower()) not in {'h', 's'}:
        print("Only enter H or S")
    return action, []

def get_dealer_action(hand, logs):
    total_points = score_hand(hand)
    if total_points >= 17:
        return 's', logs
    return 'h', logs

def initial_state():
    deck = new_shuffled_deck()
    players = ['Player', 'Dealer']
    hands = {player: [] for player in players}
    actions = {'Player': get_player_action, 'Dealer': get_dealer_action}
    return deck, players, hands, actions

def process_game():
    update_hands = update_dict
    logs = []
    deck, players, hands, actions = initial_state()
    turns = cycle(players)
    while players:
        turn = next(turns)
        hand = hands[turn]
        logs = [*logs, f'\n> {turn} hand is: {" ".join(str(card) for card in hand)}']
        action, logs = actions[turn](hand, logs)
        match action:
            case 'h':
                logs = [*logs, f'Hit']
                hand, deck = draw_to_hand(hand, deck)
                hands = update_hands(hands, turn, hand)
                logs = [*logs, f'Drawn {hand[-1]}']
                try:
                    total = score_hand(hand)
                    logs = [*logs, f'Current score is {total}']
                except ValueError:
                    logs = [*logs, f'BUSTED']
                    return hands, logs
                hands = update_dict(hands, turn, hand)
            case 's':
                logs = [*logs, f'Stay']
                players = [player for player in players if player != turn]
                turns = cycle(players)
    return hands, logs

def get_score(hand):
    try:
        total = score_hand(hand)
        return total
    except ValueError:
        return 'BUSTED'

def end_game(state):
    hands, logs = state
    logs = [*logs, f'\n*** Game Over ***']
    for player, hand in hands.items():
        logs = [*logs, f'{player}:{get_score(hand)}']
    return logs

run = compose2(compose2(process_game, end_game), stream_logs)
run()
