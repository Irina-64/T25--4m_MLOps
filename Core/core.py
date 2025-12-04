# Core/core.py
# Реализация Подкидного дурака (минимально необходимая логика) в процедурно-ООП стиле.
# Поддерживает:
# - 36-карточную колоду: 6,7,8,9,10,J,Q,K,A
# - козырь определяется при раздаче
# - до 6 карт в атаке (правило: максимум 6 на одного защитника)
# - подбрасывать разрешено всем атакующим (по рангу карт на столе)
# - базовый игровой цикл с API для ML: get_state(), legal_actions(player_id), step(action)
#
# ВАЖНО: я выбрал одну из распространённых варинтов очередности:
#  - На каждом ходу есть attacker (атакующий) и defender (следующий по кругу).
#  - Если защитник отбился (все карты отбиты) — карты уносятся в сброс, следующим атакующим
#    становится игрок слева от предыдущего атакующего (т. е. ход переходит дальше по кругу).
#  - Если защитник взял (take) — он берет все карты со стола, и следующим атакующим
#    становится игрок слева от взявшего (т.е. ход переходит за дефендера).
# Эти детали можно изменить в методах _on_successful_defense и _on_take.
#
# Для ML важно: state - словарь с рукою игрока, верхним козырём, картами на столе, картами на руке других игроков
# (опционально), и т.д. legal_actions возвращает список возможных действий в текущем состоянии.

# (модифицирована поддержка defend: принимаем и ('defend', attack_index, card) и ('defend', attack_card, card))
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["♣", "♦", "♥", "♠"]
MAX_ATTACK_CARDS = 6


class Card:
    __slots__ = ("rank", "suit")

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def rank_index(self):
        return RANKS.index(self.rank)

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return (
            isinstance(other, Card)
            and self.rank == other.rank
            and self.suit == other.suit
        )

    def __hash__(self):
        return hash((self.rank, self.suit))


class Deck:
    def card_value(self, card: Card, trump_suit: Optional[str] = None) -> int:
        rank_value = RANKS.index(card.rank)
        if trump_suit and card.suit == trump_suit:
            return 100 + rank_value
        return rank_value

    def __init__(self):
        self.cards: List[Card] = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(self.cards)

    def draw(self, n=1) -> List[Card]:
        drawn = []
        for _ in range(n):
            if self.cards:
                drawn.append(self.cards.pop())
            else:
                break
        return drawn

    def top_trump(self) -> Optional[Card]:
        return self.cards[0] if self.cards else None

    def __len__(self):
        return len(self.cards)


class Player:
    def __init__(self, name: str, pid: int):
        self.name = name
        self.id = pid
        self.hand: List[Card] = []
        self.hand_history: List[Card] = []

    def receive(self, cards: List[Card]):
        self.hand.extend(cards)

    def remove(self, card: Card):
        for i, c in enumerate(self.hand):
            if c == card:
                removed = self.hand.pop(i)
                self.hand_history.append(removed)  # <-- сохраняем в историю
                return removed
        raise ValueError("card not in hand")

    def has_card(self, card: Card) -> bool:
        return any(c == card for c in self.hand)

    def sort_hand(self, trump_suit: Optional[str] = None):
        def key(c: Card):
            trump_key = 0 if (trump_suit and c.suit == trump_suit) else 1
            return (trump_key, c.suit, c.rank_index())

        self.hand.sort(key=key)

    def __repr__(self):
        return f"Player({self.name}, id={self.id}, hand={self.hand})"


class DurakGame:
    def __init__(self, player_names: List[str]):
        if not (2 <= len(player_names) <= 6):
            raise ValueError("Поддерживаются 2-6 игроков")
        self.players: List[Player] = [Player(n, i) for i, n in enumerate(player_names)]
        self.n = len(self.players)
        self.deck = Deck()
        self.trump_card = self.deck.top_trump()
        self.trump_suit = self.trump_card.suit if self.trump_card else None
        self.turn_order = deque(range(self.n))
        self.discard_pile: List[Card] = []
        self.table: List[Tuple[Card, Optional[Card]]] = []
        self.attacker = None
        self.defender = None
        self.finished = False
        self.winner_ids: List[int] = []
        self._deal_initial()
        self._init_first_attacker()

    def _deal_initial(self):
        for p in self.players:
            p.receive(self.deck.draw(6))
        for p in self.players:
            p.sort_hand(self.trump_suit)

    def _init_first_attacker(self):
        min_trump = None
        min_pid = 0
        for p in self.players:
            for c in p.hand:
                if c.suit == self.trump_suit:
                    if (min_trump is None) or (c.rank_index() < min_trump.rank_index()):
                        min_trump = c
                        min_pid = p.id
        self.turn_order = deque(range(self.n))
        while self.turn_order[0] != min_pid:
            self.turn_order.rotate(-1)
        self.attacker = self.turn_order[0]
        self.defender = self.turn_order[1 % self.n]

    def _next_turn_order(self):
        self.turn_order.rotate(-1)
        self.attacker = self.turn_order[0]
        self.defender = self.turn_order[1 % self.n]

    def _cards_on_table_count(self):
        return sum(1 for a, d in self.table if a is not None) + sum(
            1 for a, d in self.table if d is not None
        )

    def _ranks_on_table(self) -> List[str]:
        ranks = []
        for a, d in self.table:
            if a:
                ranks.append(a.rank)
            if d:
                ranks.append(d.rank)
        return ranks

    def _refill_hands(self):
        for pid in list(self.turn_order):
            player = self.players[pid]
            need = max(0, 6 - len(player.hand))
            if need:
                drawn = self.deck.draw(need)
                player.receive(drawn)
                player.sort_hand(self.trump_suit)

    def can_beat(self, attack_card: Card, defense_card: Card) -> bool:
        # same suit and higher rank
        if defense_card.suit == attack_card.suit:
            return defense_card.rank_index() > attack_card.rank_index()
        # trump beats non-trump
        if defense_card.suit == self.trump_suit and attack_card.suit != self.trump_suit:
            return True
        return False

    # Actions: ('attack', Card), ('defend', attack_index_or_card, Card),
    # ('add', Card), ('take',), ('pass',)
    def legal_actions(self, pid: int) -> List[Any]:
        if self.finished:
            return []
        pid = int(pid)
        if (
            pid != self.attacker
            and pid != self.defender
            and pid not in self._other_attackers()
        ):
            return []
        actions = []
        player = self.players[pid]

        # Attacker actions
        if pid == self.attacker:
            if not self.table:
                for c in player.hand:
                    actions.append(("attack", c))
            else:
                ranks_present = set(self._ranks_on_table())
                if self._cards_on_table_count() < MAX_ATTACK_CARDS:
                    for c in player.hand:
                        if c.rank in ranks_present:
                            actions.append(("add", c))
            # attacker may finish attack
            actions.append(("pass",))

        # Other attackers (can add)
        if pid in self._other_attackers():
            if self.table and self._cards_on_table_count() < MAX_ATTACK_CARDS:
                ranks_present = set(self._ranks_on_table())
                for c in player.hand:
                    if c.rank in ranks_present:
                        actions.append(("add", c))
                actions.append(("pass",))
            else:
                actions.append(("pass",))

        # Defender actions: for each unprotected attack provide defend options
        if pid == self.defender:
            for i, (a, d) in enumerate(self.table):
                if d is None:
                    for c in player.hand:
                        if self.can_beat(a, c):
                            # Provide defend action in TWO equivalent forms for convenience:
                            # ('defend', attack_card, defend_card)  and  ('defend', attack_index, defend_card)
                            actions.append(("defend", a, c))
                            actions.append(("defend", i, c))
            actions.append(("take",))
        return actions

    def _other_attackers(self) -> List[int]:
        return [
            pid
            for pid in range(self.n)
            if pid != self.attacker
            and pid != self.defender
            and len(self.players[pid].hand) > 0
        ]

    def step(self, pid: int, action: Tuple) -> Dict[str, Any]:
        if self.finished:
            return {
                "ok": False,
                "message": "Game finished",
                "state": self.get_state(pid),
            }

        legal = self.legal_actions(pid)
        # Normalize defend action in legal for matching: if legal contains ('defend', Card, c),
        # and incoming action may use index, or vice versa. We'll accept both forms.
        if not self._action_is_legal(action, legal):
            return {
                "ok": False,
                "message": f"Illegal action {action}",
                "state": self.get_state(pid),
            }

        typ = action[0]
        if typ == "attack":
            _, card = action
            self._do_attack(pid, card)
            return {"ok": True, "message": "attack made", "state": self.get_state(pid)}
        if typ == "add":
            _, card = action
            self._do_add(pid, card)
            return {"ok": True, "message": "card added", "state": self.get_state(pid)}
        if typ == "defend":
            # action can be ('defend', attack_index, card) OR ('defend', attack_card, card)
            _, attack_ident, card = action
            self._do_defend(pid, attack_ident, card)
            return {"ok": True, "message": "defended", "state": self.get_state(pid)}
        if typ == "take":
            self._do_take(pid)
            self._refill_hands()
            self._check_finishers()
            return {"ok": True, "message": "took cards", "state": self.get_state(pid)}
        if typ == "pass":
            self._on_pass()
            self._refill_hands()
            self._check_finishers()
            return {"ok": True, "message": "pass", "state": self.get_state(pid)}

        return {"ok": False, "message": "unknown action", "state": self.get_state(pid)}

    def _action_is_legal(self, action: Tuple, legal_list: List[Tuple]) -> bool:
        # quick path
        if action in legal_list:
            return True
        # if defend with index but legal_list has defend with card (or vice versa), compare semantically
        if action[0] == "defend":
            _, attack_ident, card = action
            for act in legal_list:
                if act[0] != "defend":
                    continue
                # act can be ('defend', a_card, c) or ('defend', idx, c)
                _, la, lc = act
                if lc == card:
                    # if la is int and attack_ident is Card and matches table[la], accept
                    if isinstance(la, int) and isinstance(attack_ident, Card):
                        if la < len(self.table) and self.table[la][0] == attack_ident:
                            return True
                    # if la is Card and attack_ident is int
                    if isinstance(la, Card) and isinstance(attack_ident, int):
                        if attack_ident < len(self.table) and self.table[attack_ident][0] == la:
                            return True
                    # if both Cards: compare
                    if isinstance(la, Card) and isinstance(attack_ident, Card) and la == attack_ident:
                        return True
                    # if both ints and equal
                    if isinstance(la, int) and isinstance(attack_ident, int) and la == attack_ident:
                        return True
            return False
        return False

    def _do_attack(self, pid: int, card: Card):
        player = self.players[pid]
        if not player.has_card(card):
            raise ValueError("Player does not have this card")
        player.remove(card)
        self.table.append((card, None))

    def _do_add(self, pid: int, card: Card):
        player = self.players[pid]
        if not player.has_card(card):
            raise ValueError("Player does not have this card")
        player.remove(card)
        self.table.append((card, None))

    def _do_defend(self, pid: int, attack_ident: Any, card: Card):
        # attack_ident may be int index or Card object
        # find index
        if isinstance(attack_ident, int):
            attack_index = attack_ident
        else:
            # find first table index with attack card equal to attack_ident and not yet defended
            attack_index = None
            for i, (a, d) in enumerate(self.table):
                if a == attack_ident and d is None:
                    attack_index = i
                    break
            if attack_index is None:
                raise IndexError("attack card not found or already defended")

        if attack_index < 0 or attack_index >= len(self.table):
            raise IndexError("attack_index out of bounds")
        a, d = self.table[attack_index]
        if d is not None:
            raise ValueError("Already defended")
        player = self.players[pid]
        if not player.has_card(card):
            raise ValueError("No such card in hand")
        if not self.can_beat(a, card):
            raise ValueError("Cannot beat")
        player.remove(card)
        self.table[attack_index] = (a, card)
        # do not auto-end round here; that remains up to attackers/pass

    def _do_take(self, pid: int):
        taken_cards = []
        for a, d in self.table:
            if a:
                taken_cards.append(a)
            if d:
                taken_cards.append(d)
        self.table.clear()
        self.players[pid].receive(taken_cards)
        self.players[pid].sort_hand(self.trump_suit)
        # rotate queue so next attacker is left of taker
        while self.turn_order[0] != pid:
            self.turn_order.rotate(-1)
        self.turn_order.rotate(-1)
        self.attacker = self.turn_order[0]
        self.defender = self.turn_order[1 % self.n]

    def _on_pass(self):
        any_unprotected = any(d is None for a, d in self.table)
        if any_unprotected:
            self._do_take(self.defender)
            return
        for a, d in self.table:
            if a:
                self.discard_pile.append(a)
            if d:
                self.discard_pile.append(d)
        self.table.clear()
        self._next_turn_order()

    def _check_finishers(self):
        active = [p for p in self.players if len(p.hand) > 0]
        if len(active) <= 1:
            self.finished = True
            self.winner_ids = [p.id for p in self.players if len(p.hand) == 0]
            return

    def get_state(self, pid: Optional[int] = None) -> Dict[str, Any]:
        state = {}
        state["trump_suit"] = self.trump_suit
        state["deck_count"] = len(self.deck)
        state["discard_count"] = len(self.discard_pile)
        state["table"] = [
            (a.__repr__() if a else None, d.__repr__() if d else None)
            for a, d in self.table
        ]
        state["attacker"] = self.attacker
        state["defender"] = self.defender
        state["turn_order"] = list(self.turn_order)
        state["finished"] = self.finished
        state["winners"] = self.winner_ids
        state["hand_sizes"] = {p.id: len(p.hand) for p in self.players}
        state["hand_history"] = {p.id: p.hand_history for p in self.players}

        if pid is not None:
            p = self.players[pid]
            state["your_hand"] = [c.__repr__() for c in p.hand]
        else:
            state["your_hand"] = None
        return state

    def pretty_print(self):
        print(
            f"Trump: {self.trump_suit} | Deck: {len(self.deck)} | Discard: {len(self.discard_pile)}"
        )
        for p in self.players:
            print(f"Player {p.id} {p.name}: {len(p.hand)} cards - {p.hand}")
        print("Table:")
        for i, (a, d) in enumerate(self.table):
            print(f"  {i}: {a} -> {d}")
        print("Turn order:", list(self.turn_order))
        print("Attacker:", self.attacker, "Defender:", self.defender)
        print("---")


def random_agent_actions(game: DurakGame, pid: int):
    legal = game.legal_actions(pid)
    if not legal:
        return None
    prefer = [a for a in legal if a[0] in ("attack", "defend", "add")]
    choices = prefer if prefer else legal
    return random.choice(choices)


def demo_random_play(names=["A", "B", "C", "D"]):
    g = DurakGame(names)
    print("Start game")
    g.pretty_print()
    steps = 0
    while not g.finished and steps < 500:
        current_players = list(g.turn_order)
        acted = False
        for pid in current_players:
            legal = g.legal_actions(pid)
            if not legal:
                continue
            act = random_agent_actions(g, pid)
            if act is None:
                continue
            res = g.step(pid, act)
            acted = True
            print(f"Player {pid} action: {act} -> {res['message']}")
            g.pretty_print()
            steps += 1
            if g.finished or steps >= 500:
                break
        if not acted:
            g.step(g.attacker, ("pass",))
    print("Game finished:", g.finished, "Winners:", g.winner_ids)
    return g


if __name__ == "__main__":
    demo_random_play(["Иван", "Оля", "Петя", "Маша"])
