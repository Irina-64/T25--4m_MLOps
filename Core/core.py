# Core/durak.py
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


import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["♣", "♦", "♥", "♠"]  # или 'clubs','diamonds','hearts','spades'
MAX_ATTACK_CARDS = 6  # макс карт в одной атаке на защитника


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
            return 100 + rank_value  # козыри сильнее
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
        # сортировка: сначала по тому, козырь или нет, затем по рангу
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
        # turn order as deque of player indices
        self.turn_order = deque(range(self.n))
        self.discard_pile: List[Card] = []
        # table: list of pairs (attack_card, defend_card_or_None)
        self.table: List[Tuple[Card, Optional[Card]]] = []
        # current attacker index = turn_order[0], defender = turn_order[1]
        self.attacker = None
        self.defender = None
        # game ended ?
        self.finished = False
        self.winner_ids: List[int] = []
        # initialize
        self._deal_initial()
        self._init_first_attacker()

    def _deal_initial(self):
        # каждому по 6 карт
        for p in self.players:
            p.receive(self.deck.draw(6))
        # после раздачи определяем козырь (верхняя карта остаётся внизу колоды,
        # смотрим на самую нижнюю карту в стандартной реализации — deck.cards[0])
        # но в нашей Deck.top_trump это deck.cards[0] (если deck.cards не пуст)
        for p in self.players:
            p.sort_hand(self.trump_suit)

    def _init_first_attacker(self):
        # общепринято: первым атакует тот, у кого самая низкая козырная карта
        min_trump = None
        min_pid = 0
        for p in self.players:
            for c in p.hand:
                if c.suit == self.trump_suit:
                    if (min_trump is None) or (c.rank_index() < min_trump.rank_index()):
                        min_trump = c
                        min_pid = p.id
        # если нет козырей у никого — первый игрок (player 0)
        self.turn_order = deque(range(self.n))
        # rotate so that min_pid is first
        while self.turn_order[0] != min_pid:
            self.turn_order.rotate(-1)
        self.attacker = self.turn_order[0]
        self.defender = self.turn_order[1 % self.n]

    # ---------- Utility ----------
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
        # каждый добирает карты до 6, начиная с атакующего и далее по кругу,
        # но не включаем игроков, у которых уже 0 карт (они вышли).
        # Стандарт: добор до 6 в порядке очереди начиная с атакующего.
        for pid in list(self.turn_order):
            player = self.players[pid]
            need = max(0, 6 - len(player.hand))
            if need:
                drawn = self.deck.draw(need)
                player.receive(drawn)
                player.sort_hand(self.trump_suit)

    # ---------- Core rules ----------
    def can_beat(self, attack_card: Card, defense_card: Card) -> bool:
        # защита возможна, если:
        # - та же масть и старше по рангу, или
        # - defense_card масть = trump и attack_card не trump
        if defense_card.suit == attack_card.suit:
            return defense_card.rank_index() > attack_card.rank_index()
        if defense_card.suit == self.trump_suit and attack_card.suit != self.trump_suit:
            return True
        return False

    # Actions: 'attack', 'defend', 'add' (подброс), 'take', 'pass' (закончить подбрасывать)
    # Represent an action as tuple: ('attack', Card), ('defend', attack_index, Card),
    # ('add', Card), ('take',), ('pass',)
    def legal_actions(self, pid: int) -> List[Any]:
        """Возвращает список легальных действий для игрока pid в текущем состоянии."""
        if self.finished:
            return []
        pid = int(pid)
        if (
            pid != self.attacker
            and pid != self.defender
            and pid not in self._other_attackers()
        ):
            # игрок может только подбрасывать в пределах очереди, иначе не участвует
            return []
        actions = []
        player = self.players[pid]

        # если игрок — текущий атакующий и еще нет атак на столе (новая атака) или можно подбрасывать:
        if pid == self.attacker:
            # если в таблице нет атакующих карт (начало раунда): он может положить любую карту
            if not self.table:
                for c in player.hand:
                    actions.append(("attack", c))
            else:
                # он может подбросить, но только карты с рангом, который уже есть на столе,
                # и если лимит MAX_ATTACK_CARDS не превышен (исходя из количества атакующих карт)
                ranks_present = set(self._ranks_on_table())
                if self._cards_on_table_count() < MAX_ATTACK_CARDS:
                    for c in player.hand:
                        if c.rank in ranks_present:
                            actions.append(("add", c))
            # также может pass (закончить атаку) если он решил не добавлять
            actions.append(("pass",))

        # другие игроки (кроме защитника) могут подбрасывать после первой атаки,
        # но только если уже есть атаки и лимит не превышен
        if pid in self._other_attackers():
            if self.table and self._cards_on_table_count() < MAX_ATTACK_CARDS:
                ranks_present = set(self._ranks_on_table())
                for c in player.hand:
                    if c.rank in ranks_present:
                        actions.append(("add", c))
                actions.append(("pass",))
            else:
                actions.append(("pass",))

        # защитник может:
        if pid == self.defender:
            # defend: для каждой незащищённой атакующей карты предложить взбивание
            for i, (a, d) in enumerate(self.table):
                if d is None:
                    for c in player.hand:
                        if self.can_beat(a, c):
                            actions.append(("defend", i, c))
            # take всегда доступен (взять все)
            actions.append(("take",))
        return actions

    def _other_attackers(self) -> List[int]:
        # игроки кроме attacker и defender, которые находятся в очереди между defender+1 ... attacker-1
        # На деле все остальные игроки могут подбрасывать, но очередность подбрасывания идет по кругу от атакующего.
        # Для упрощения: считаем, что все остальные игроки могут подбрасывать (они будут выбирать 'add' или 'pass').
        return [
            pid
            for pid in range(self.n)
            if pid != self.attacker
            and pid != self.defender
            and len(self.players[pid].hand) > 0
        ]

    def step(self, pid: int, action: Tuple) -> Dict[str, Any]:
        """
        Применить действие и обновить состояние.
        Возвращает dict с info: {'ok':bool, 'message':str, 'state':...}
        """
        if self.finished:
            return {
                "ok": False,
                "message": "Game finished",
                "state": self.get_state(pid),
            }

        legal = self.legal_actions(pid)
        # Сравнивать действия по структуре; карточки — объекты, поэтому проверяем равенство
        if action not in legal:
            return {
                "ok": False,
                "message": f"Illegal action {action}",
                "state": self.get_state(pid),
            }

        # Обработка
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
            _, attack_index, card = action
            self._do_defend(pid, attack_index, card)
            return {"ok": True, "message": "defended", "state": self.get_state(pid)}
        if typ == "take":
            self._do_take(pid)
            # refill hands and check winners
            self._refill_hands()
            self._check_finishers()
            return {"ok": True, "message": "took cards", "state": self.get_state(pid)}
        if typ == "pass":
            # "pass" — окончание подбрасывания атакующими: завершение раунда
            self._on_pass()
            # refill hands and check winners
            self._refill_hands()
            self._check_finishers()
            return {"ok": True, "message": "pass", "state": self.get_state(pid)}

        return {"ok": False, "message": "unknown action", "state": self.get_state(pid)}

    # ---------- Action implementations ----------
    def _do_attack(self, pid: int, card: Card):
        player = self.players[pid]
        if not player.has_card(card):
            raise ValueError("Player does not have this card")
        player.remove(card)
        self.table.append((card, None))

    def _do_add(self, pid: int, card: Card):
        # подбрасывание: кладём новую атаку (с defend=None)
        player = self.players[pid]
        if not player.has_card(card):
            raise ValueError("Player does not have this card")
        player.remove(card)
        self.table.append((card, None))

    def _do_defend(self, pid: int, attack_index: int, card: Card):
        # защитник кладёт карту на конкретную атаку (attack_index)
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
        # если все атаки застабилизированы (все защищены, и атакующие не могут/не хотят подбрасывать), то
        # оборона успешна -> окончание защиты. Но мы не авто-оканчиваем здесь, ожидаем pass от атакующих.
        # Эта реализация оставляет окончание раунда на 'pass' от атакующих.

    def _do_take(self, pid: int):
        # defender берет все карты со стола (и уничтожается таблица)
        # defender получает все карты (attack and defense)
        taken_cards = []
        for a, d in self.table:
            if a:
                taken_cards.append(a)
            if d:
                taken_cards.append(d)
        self.table.clear()
        self.players[pid].receive(taken_cards)
        self.players[pid].sort_hand(self.trump_suit)
        # После взятия: ход переходит на игрока слева от взявшего
        # реализуем как поворот очереди так, чтобы следующий атакующий был слева от взявшего
        # Найдем позицию взявшего в текущем turn_order и сделаем его +1 первым
        while self.turn_order[0] != pid:
            self.turn_order.rotate(-1)
        # теперь очередь начинается с взявшего; следующий атакующий — следующий после взявшего
        self.turn_order.rotate(-1)
        self.attacker = self.turn_order[0]
        self.defender = self.turn_order[1 % self.n]

    def _on_pass(self):
        # Когда атакующие объявляют 'pass' — защита успешна (если все атаки защищены),
        # или если есть незащищённые атаки — это означает, что защитник не отбился и должен взять.
        # В нашей упрощённой логике: если есть хоть одна незащищённая карта -> то defender must take.
        any_unprotected = any(d is None for a, d in self.table)
        if any_unprotected:
            # defender берет
            self._do_take(self.defender)
            return
        # все защищено — отправляем все карты в discard
        for a, d in self.table:
            if a:
                self.discard_pile.append(a)
            if d:
                self.discard_pile.append(d)
        self.table.clear()
        # next attacker: игрок слева от текущего атакующего
        # т. е. просто продвинем очередь, чтобы предыдущий атакующий ушёл в конец
        self._next_turn_order()

    def _check_finishers(self):
        # игрока с 0 карт — он вышел (порядок выхода может пригодиться для побед)
        # если осталось <=1 игрока с картами — игра окончена
        active = [p for p in self.players if len(p.hand) > 0]
        if len(active) <= 1:
            self.finished = True
            self.winner_ids = [p.id for p in self.players if len(p.hand) == 0]
            # последний оставшийся — дурак (если есть один)
            return

    # ---------- Observability for ML ----------
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
        
        # hand_history для RewardSystem
        state["hand_history"] = {p.id: p.hand_history for p in self.players}
        
        if pid is not None:
            p = self.players[pid]
            state["your_hand"] = [c.__repr__() for c in p.hand]
        else:
            state["your_hand"] = None
        return state

    # ---------- Helpers for demo ----------
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


# ----------------- Simple interactive demo / random-play bot -----------------


def random_agent_actions(game: DurakGame, pid: int):
    """Простейший агент: выбирает случайное из legal_actions."""
    legal = game.legal_actions(pid)
    if not legal:
        return None
    # prefer attacks/defends over pass/take
    prefer = [a for a in legal if a[0] in ("attack", "defend", "add")]
    choices = prefer if prefer else legal
    return random.choice(choices)


def demo_random_play(names=["A", "B", "C", "D"]):
    g = DurakGame(names)
    print("Start game")
    g.pretty_print()
    steps = 0
    # loop until finished or steps limit
    while not g.finished and steps < 500:
        current_players = list(g.turn_order)
        # each player in turn_order may act depending on allowed actions; to simplify,
        # we loop over all players and let them act if they have legal actions that change table.
        # Real game flow more sequential, but for demo this is acceptable.
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
            # print step
            print(f"Player {pid} action: {act} -> {res['message']}")
            # if table changed or round ended, print
            g.pretty_print()
            steps += 1
            if g.finished or steps >= 500:
                break
        if not acted:
            # nobody сделал ход — этот раунд завершаем
            # вызываем pass от attacker
            g.step(g.attacker, ("pass",))
    print("Game finished:", g.finished, "Winners:", g.winner_ids)
    return g


# ---------- If run as script, демонстрация ----------
if __name__ == "__main__":
    demo_random_play(["Иван", "Оля", "Петя", "Маша"])
